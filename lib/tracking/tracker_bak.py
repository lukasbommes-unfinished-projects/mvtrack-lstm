import uuid
import torch
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
        self.first_init_call = False
        self.mvs_residuals_seq = deque(maxlen=seq_len)#torch.zeros(1, self.seq_len, 5, 600, 1000)
        self.boxes_seq = deque(maxlen=seq_len+1)#torch.zeros(1, self.seq_len, 1, 5)
        #self.last_mvs_residuals = torch.zeros(size=(1, 5, 600, 1000))

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
        detection_boxes[idx]
        return detection_boxes[idx], detection_scores[idx]


    def init(self, mvs_residuals, detection_boxes, detection_scores):
        # remove detections with confidence lower than det_conf_threshold
        if self.det_conf_threshold is not None:
            detection_boxes, detection_scores = self._filter_low_confidence_detections(detection_boxes, detection_scores)

        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)

        # handle matches by incremeting the counter for redetection and resetting the one for lost
        for d, t in matches:
            self.boxes[t] = detection_boxes[d]

        # handle unmatched detections by spawning new trackers in pending state
        for d in unmatched_detectors:
            if self.use_numeric_ids:
                self.box_ids.append(self.next_id)
                self.next_id += 1
            else:
                uid = uuid.uuid4()
                self.box_ids.append(uid)
            self.boxes = np.vstack((self.boxes, detection_boxes[d]))

        # handle unmatched tracker predictions by counting how often a target got lost subsequently
        for t in unmatched_trackers:
            self.boxes = np.delete(self.boxes, t, axis=0)
            self.box_ids.pop(t)

        # store boxes and mvs_residuals for later use with LSTM, skip first mvs_residual
        self.boxes_seq.append(np.copy(self.boxes))
        if self.first_init_call:
            self.mvs_residuals_seq.append(np.copy(mvs_residuals))
        self.first_init_call = True

        print("Init")
        print("mvs", self.mvs_residuals_seq)
        print("boxes", self.boxes_seq)


    def update(self, mvs_residuals, detection_boxes, detection_scores):
        if self.measure_timing:
            start_update = torch.cuda.Event(enable_timing=True)
            end_update = torch.cuda.Event(enable_timing=True)
            start_update.record()

        # remove detections with confidence lower than det_conf_threshold
        if self.det_conf_threshold is not None:
            detection_boxes, detection_scores = self._filter_low_confidence_detections(detection_boxes, detection_scores)

        # bring boxes into next state
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

        print("Update")
        print("mvs", self.mvs_residuals_seq)
        print("boxes", self.boxes_seq)


    # def preprocess_boxes_(self, boxes):
    #
    #
    # def preprocess_mvs_residuals(self, mvs_residuals):
    #
    #     # perform preprocessing steps
    #     mvs_residuals = torch.from_numpy(mvs_residuals).type(torch.float).unsqueeze(0)
    #     mvs_residuals = mvs_residuals.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    #
    #     # check length of mvs_residuals in seq dimension
    #
    #     # if length equals seq_len, remove oldest mvs_residuals and attach a new one


    def predict(self, mvs_residuals, save_last_boxes=True):
        if self.measure_timing:
            start_predict = torch.cuda.Event(enable_timing=True)
            end_predict = torch.cuda.Event(enable_timing=True)
            start_predict.record()

        # save mvs_residuals for later steps as input to LSTM
        self.mvs_residuals_seq.append(np.copy(mvs_residuals))

        # if there are no boxes skip prediction step
        #if np.shape(self.boxes)[0] == 0:
        #    return

        # TODO: call preprocessing function to enqueue new boxes and motion vectors
        # into the

        # check if frame is not a key frame
        if bool(np.sum(mvs_residuals)):
            mvs_residuals = torch.from_numpy(mvs_residuals).type(torch.float).unsqueeze(0)
            mvs_residuals = mvs_residuals.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            self.last_mvs_residuals = mvs_residuals

        # pre process boxes
        boxes_prev = np.copy(self.boxes)
        boxes_prev = torch.from_numpy(boxes_prev)
        num_boxes = (boxes_prev.shape)[0]

        # insert batch index
        boxes_prev_ = boxes_prev.clone()
        boxes_prev_tmp = torch.zeros(num_boxes, 5).float()
        boxes_prev_tmp[:, 1:] = boxes_prev_
        boxes_prev_ = boxes_prev_tmp
        boxes_prev_ = boxes_prev_.unsqueeze(0)  # add batch dimension

        # change box format to [frame_idx, x1, x2, y1, y2]
        boxes_prev_[..., -2] = boxes_prev_[..., -2] + boxes_prev_[..., -4]
        boxes_prev_[..., -1] = boxes_prev_[..., -1] + boxes_prev_[..., -3]

        boxes_prev_ = boxes_prev_.to(self.device)
        mvs_residuals = self.last_mvs_residuals
        mvs_residuals = mvs_residuals.to(self.device)

        # feed into model, retrieve output
        with torch.set_grad_enabled(False):
            #torch.stack([torch.from_numpy(boxes).float() for boxes in boxes]).shape
            if self.measure_timing:
                start_inference = torch.cuda.Event(enable_timing=True)
                end_inference = torch.cuda.Event(enable_timing=True)
                start_inference.record()
            velocities_pred = self.model(mvs_residuals, boxes_prev_)
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

        boxes_pred = bbox_transform_inv_otcd(boxes=boxes_prev_[..., 1:].cpu(), deltas=velocities_pred, sigma=1.5).squeeze().numpy()
        # change box format to [xmin, ymin, w, h]
        boxes_pred[..., -2] = boxes_pred[..., -2] - boxes_pred[..., -4]
        boxes_pred[..., -1] = boxes_pred[..., -1] - boxes_pred[..., -3]
        self.boxes = boxes_pred

        # save boxes for later steps as input to LSTM
        if save_last_boxes:
            self.boxes_seq.append(np.copy(self.boxes))

            print("Predict")
            print("mvs", self.mvs_residuals_seq)
            print("boxes", self.boxes_seq)

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
