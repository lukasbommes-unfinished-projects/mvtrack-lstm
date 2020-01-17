import uuid
import torch
import copy
import numpy as np
import cv2
from collections import deque

from lib.model.tracknet import TrackNet
from lib.tracking.utils import match_bounding_boxes
from lib.dataset.velocities import bbox_transform_inv_otcd
from lib.utils import load_pretrained_weights


class TrackerMVLSTM:
    def __init__(self, iou_threshold, det_conf_threshold, state_thresholds,
        seq_len, weights_file, device=None, use_numeric_ids=False,
        measure_timing=False):
        self.iou_threshold = iou_threshold
        self.det_conf_threshold = det_conf_threshold
        self.seq_len = seq_len
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.use_numeric_ids = use_numeric_ids

        self.bbox_reg_mean = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.bbox_reg_std = torch.tensor([0.1, 0.1, 0.2, 0.2])

        self.state_counters = {"missed": [], "redetected": []}
        self.target_states = []

        # target state transition thresholds
        self.pending_to_confirmed_thres = state_thresholds[0]
        self.confirmed_to_pending_thres = state_thresholds[1]
        self.pending_to_deleted_thres = state_thresholds[2]

        self.boxes = np.empty(shape=(0, 4))
        self.box_ids = []
        self.next_id = 1

        # cor storing sequence of subsequent mvs_residuals and boxes for use with LSTM
        self.mvs_residuals_seq = deque(maxlen=seq_len)
        self.boxes_seq = deque(maxlen=seq_len)
        self.boxes_init = np.empty(shape=(0, 4))

        self.model = TrackNet()
        self.model = self.model.to(self.device)
        self.model.device = self.device
        self.model = load_pretrained_weights(self.model, weights_file)
        self.model.eval()

        # for timing analaysis
        self.measure_timing = measure_timing
        self.last_inference_dt = np.inf
        self.last_predict_dt = np.inf
        self.last_update_dt = np.inf


    def _filter_low_confidence_detections(self, detection_boxes, detection_scores):
        idx = np.nonzero(detection_scores >= self.det_conf_threshold)
        return detection_boxes[idx], detection_scores[idx]


    def init(self, mvs_residuals, detection_boxes, detection_scores):
        # remove detections with confidence lower than det_conf_threshold
        if self.det_conf_threshold is not None:
            detection_boxes, detection_scores = self._filter_low_confidence_detections(detection_boxes, detection_scores)

        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = match_bounding_boxes(self.boxes_init, detection_boxes, self.iou_threshold)

        # handle matches by incremeting the counter for redetection and resetting the one for lost
        for d, t in matches:
            self.boxes_init[t] = detection_boxes[d]

        # handle unmatched detections by spawning new trackers in pending state
        for d in unmatched_detectors:
            self.boxes_init = np.vstack((self.boxes_init, detection_boxes[d]))

        # handle unmatched tracker predictions by counting how often a target got lost subsequently
        for t in unmatched_trackers:
            self.boxes_init = np.delete(self.boxes_init, t, axis=0)

        # store boxes and mvs_residuals for later use with LSTM, skip first mvs_residual
        self.boxes_seq.append(np.copy(self.boxes_init))
        self.mvs_residuals_seq.append(np.copy(mvs_residuals))

        #print("Init")
        #print("boxes", self.boxes_seq)


    def update(self, mvs_residuals, detection_boxes, detection_scores):
        if self.measure_timing:
            start_update = torch.cuda.Event(enable_timing=True)
            end_update = torch.cuda.Event(enable_timing=True)
            start_update.record()

        # remove detections with confidence lower than det_conf_threshold
        if self.det_conf_threshold is not None:
            detection_boxes, detection_scores = self._filter_low_confidence_detections(detection_boxes, detection_scores)

        # bring boxes into next state
        if np.shape(self.boxes)[0] > 0:
            self.predict(mvs_residuals, save_last_boxes=False)  # we want to save boxes only after association

        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)

        # handle matches by incremeting the counter for redetection and resetting the one for lost
        for d, t in matches:
            self.state_counters["missed"][t] = 0  # reset lost counter
            self.state_counters["redetected"][t] += 1  # increment redetection counter
            self.boxes[t] = detection_boxes[d]
            # update target state based on counter values
            if self.state_counters["redetected"][t] >= self.pending_to_confirmed_thres:
                self.target_states[t] = "confirmed"

        # handle unmatched detections by spawning new trackers in pending state
        for d in unmatched_detectors:
            self.state_counters["missed"].append(0)
            self.state_counters["redetected"].append(0)
            if self.pending_to_confirmed_thres > 0:
                self.target_states.append("pending")
            elif self.pending_to_confirmed_thres == 0:
                self.target_states.append("confirmed")
            if self.use_numeric_ids:
                self.box_ids.append(self.next_id)
                self.next_id += 1
            else:
                uid = uuid.uuid4()
                self.box_ids.append(uid)
            self.boxes = np.vstack((self.boxes, detection_boxes[d]))

        # handle unmatched tracker predictions by counting how often a target got lost subsequently
        for t in unmatched_trackers:
            self.state_counters["missed"][t] += 1
            self.state_counters["redetected"][t] = 0
            # if target is not redetected for confirmed_to_pending_thres cosecutive times set its state to pending
            if self.state_counters["missed"][t] > self.confirmed_to_pending_thres:
                self.target_states[t] = "pending"
            #   if target is not redetected for pending_to_deleted_thres cosecutive times delete it
            if self.state_counters["missed"][t] > self.pending_to_deleted_thres:
                self.boxes = np.delete(self.boxes, t, axis=0)
                self.box_ids.pop(t)
                self.state_counters["missed"].pop(t)
                self.state_counters["redetected"].pop(t)
                self.target_states.pop(t)

        self.boxes_seq.append(np.copy(self.boxes))

        if self.measure_timing:
            end_update.record()
            torch.cuda.synchronize()
            self.last_update_dt = start_update.elapsed_time(end_update) / 1000.0

        #print("Update")
        #print("boxes", self.boxes_seq)


    def predict(self, mvs_residuals, save_last_boxes=True):
        if self.measure_timing:
            start_predict = torch.cuda.Event(enable_timing=True)
            end_predict = torch.cuda.Event(enable_timing=True)
            start_predict.record()

        # save mvs_residuals for later steps as input to LSTM
        self.mvs_residuals_seq.append(np.copy(mvs_residuals))

        # preprocess stored boxes
        boxes_seq_proc = copy.copy(self.boxes_seq)
        boxes_seq_proc = list(boxes_seq_proc)
        max_num_boxes = np.max([b.shape[0] for b in boxes_seq_proc])
        boxes_to_pad = [max_num_boxes-b.shape[0] for b in boxes_seq_proc]
        boxes_seq_proc = [np.pad(b, ((0, p), (1, 0)), 'constant', constant_values=0) for b, p in zip(boxes_seq_proc, boxes_to_pad)]

        boxes_seq_proc = np.stack(boxes_seq_proc, axis=0)  # stack along first axis
        boxes_seq_proc = torch.from_numpy(boxes_seq_proc).type(torch.float).unsqueeze(0)  # add batch dimension

        # change box format to [frame_idx, x1, x2, y1, y2]
        boxes_seq_proc[..., -2] = boxes_seq_proc[..., -2] + boxes_seq_proc[..., -4]
        boxes_seq_proc[..., -1] = boxes_seq_proc[..., -1] + boxes_seq_proc[..., -3]

        # preprocess stored mvs_residuals
        mvs_residuals_seq = copy.copy(self.mvs_residuals_seq)
        mvs_residuals_seq = list(mvs_residuals_seq)
        mvs_residuals_seq = np.stack(mvs_residuals_seq, axis=0)
        mvs_residuals_seq = torch.from_numpy(mvs_residuals_seq).type(torch.float).unsqueeze(0)
        mvs_residuals_seq = mvs_residuals_seq.permute(0, 1, 4, 2, 3)  # [B, S, H, W, C] -> [B, S, C, H, W]

        boxes_seq_proc = boxes_seq_proc.to(self.device)
        mvs_residuals_seq = mvs_residuals_seq.to(self.device)

        #print("max_num_boxes", max_num_boxes)

        # feed into model, retrieve output
        with torch.set_grad_enabled(False):
            #torch.stack([torch.from_numpy(boxes).float() for boxes in boxes]).shape
            if self.measure_timing:
                start_inference = torch.cuda.Event(enable_timing=True)
                end_inference = torch.cuda.Event(enable_timing=True)
                start_inference.record()
            #print("mvs_residuals_seq shape", mvs_residuals_seq.shape)
            #print("boxes_seq_proc shape", boxes_seq_proc.shape)
            velocities_pred = self.model(mvs_residuals_seq, boxes_seq_proc)
            #print("velocities_pred", velocities_pred)
            if self.measure_timing:
                end_inference.record()
                torch.cuda.synchronize()
                self.last_inference_dt = start_inference.elapsed_time(end_inference) / 1000.0

        # make sure output is on CPU
        velocities_pred = velocities_pred.cpu()
        velocities_pred = velocities_pred.view(1, -1, 4)

        # compute boxes from predicted velocities
        velocities_pred = velocities_pred.view(-1, 4) * self.bbox_reg_std + self.bbox_reg_mean
        velocities_pred = velocities_pred.view(1, -1, 4)
        velocities_pred = velocities_pred[:, 0:max_num_boxes-boxes_to_pad[-1], :]
        #print("boxes_to_pad", boxes_to_pad)

        boxes_prev = boxes_seq_proc[:, -1, 0:max_num_boxes-boxes_to_pad[-1], 1:]
        #print("boxes_prev shape", boxes_prev.shape)
        #print("velocities_pred shape", velocities_pred.shape)
        boxes_pred = bbox_transform_inv_otcd(boxes=boxes_prev.cpu(), deltas=velocities_pred, sigma=1.5, add_one=False).squeeze().numpy()
        # change box format to [xmin, ymin, w, h]
        boxes_pred[..., -2] = boxes_pred[..., -2] - boxes_pred[..., -4]
        boxes_pred[..., -1] = boxes_pred[..., -1] - boxes_pred[..., -3]
        self.boxes = boxes_pred

        #print("boxes after predict: ", self.boxes)

        # save boxes for later steps as input to LSTM
        if save_last_boxes:
            self.boxes_seq.append(np.copy(self.boxes))

            #print("Predict")
            #print("boxes", self.boxes_seq)

        if self.measure_timing:
            end_predict.record()
            torch.cuda.synchronize()
            self.last_predict_dt = start_predict.elapsed_time(end_predict) / 1000.0


    def get_boxes(self):
        # get only those boxes with state "confirmed"
        mask = [target_state == "confirmed" for target_state in self.target_states]
        boxes_filtered = self.boxes[mask]
        return boxes_filtered


    def get_box_ids(self):
        box_ids_filtered = [box_id for box_id, target_state in zip(self.box_ids, self.target_states) if target_state == "confirmed"]
        return box_ids_filtered
