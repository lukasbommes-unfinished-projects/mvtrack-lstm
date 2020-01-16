import pickle
import uuid
from collections import OrderedDict

import torch
import torchvision
import numpy as np
import cv2

import sys
sys.path.append("..")
from mvt import trackerlib

from lib.models.pnet_upsampled import PropagationNetwork as PropagationNetworkUpsampled
from lib.models.pnet_dense import PropagationNetwork as PropagationNetworkDense
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image, motion_vectors_to_grid, \
    motion_vectors_to_grid_interpolated
from lib.dataset.velocities import box_from_velocities, box_from_velocities_2d
from lib.transforms.transforms import StandardizeMotionVectors, StandardizeVelocities
from lib.utils import load_pretrained_weights


class MotionVectorTracker:
    def __init__(self, iou_threshold, det_conf_threshold,
        state_thresholds, weights_file, mvs_mode, vector_type,
        codec, stats, device=None, use_numeric_ids=False,
        measure_timing=False):
        self.iou_threshold = iou_threshold
        self.det_conf_threshold = det_conf_threshold
        self.mvs_mode = mvs_mode
        self.vector_type = vector_type
        self.codec = codec
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.use_numeric_ids = use_numeric_ids

        self.state_counters = {"missed": [], "redetected": []}
        self.target_states = []

        # target state transition thresholds
        self.pending_to_confirmed_thres = state_thresholds[0]
        self.confirmed_to_pending_thres = state_thresholds[1]
        self.pending_to_deleted_thres = state_thresholds[2]

        self.boxes = np.empty(shape=(0, 4))
        self.box_ids = []
        self.last_motion_vectors = torch.zeros(size=(1, 600, 1000, 3))
        self.next_id = 1

        self.standardize_motion_vectors = StandardizeMotionVectors(
            mean=stats.motion_vectors["mean"],
            std=stats.motion_vectors["std"])
        self.standardize_velocities = StandardizeVelocities(
            mean=stats.velocities["mean"],
            std=stats.velocities["std"],
            inverse=True)

        # load model and weigths
        if self.mvs_mode == "upsampled":
            self.model = PropagationNetworkUpsampled()
        elif self.mvs_mode == "dense":
            self.model = PropagationNetworkDense(vector_type=self.vector_type)
        self.model = self.model.to(self.device)

        self.model = load_pretrained_weights(self.model, weights_file)
        self.model.eval()

        # for timing analaysis
        self.measure_timing = measure_timing
        self.last_inference_dt = np.inf
        self.last_predict_dt = np.inf
        self.last_update_dt = np.inf


    def _preprocess_motion_vectors(self, motion_vectors, frame_shape):
        """Preprocesses motion vectors depending on the codec, vector type and mvs_mode."""
        motion_vectors_list = []
        motion_vectors = normalize_vectors(motion_vectors)
        if self.vector_type == "p+b":
            sources = ["past", "future"]
        elif self.vector_type == "p":
            sources = ["past"]
        for source in sources:
            mvs = get_vectors_by_source(motion_vectors, source)
            if self.mvs_mode == "upsampled":
                mvs = get_nonzero_vectors(mvs)
                mvs = motion_vectors_to_image(mvs, frame_shape)
            elif self.mvs_mode == "dense":
                if self.codec == "mpeg4":
                    mvs = motion_vectors_to_grid(mvs, frame_shape)
                elif self.codec == "h264":
                    mvs = motion_vectors_to_grid_interpolated(mvs, frame_shape)
            mvs = torch.from_numpy(mvs).float()
            mvs = mvs.unsqueeze(0)  # add batch dimension
            motion_vectors_list.append(mvs)
        return motion_vectors_list


    def _filter_low_confidence_detections(self, detection_boxes, detection_scores):
        idx = np.nonzero(detection_scores >= self.det_conf_threshold)
        detection_boxes[idx]
        return detection_boxes[idx], detection_scores[idx]


    def update(self, motion_vectors, frame_type, detection_boxes, detection_scores, frame_shape):
        if self.measure_timing:
            start_update = torch.cuda.Event(enable_timing=True)
            end_update = torch.cuda.Event(enable_timing=True)
            start_update.record()

        # remove detections with confidence lower than det_conf_threshold
        if self.det_conf_threshold is not None:
            detection_boxes, detection_scores = self._filter_low_confidence_detections(detection_boxes, detection_scores)

        # bring boxes into next state
        self.predict(motion_vectors, frame_type, frame_shape)

        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = trackerlib.match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)

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

        if self.measure_timing:
            end_update.record()
            torch.cuda.synchronize()
            self.last_update_dt = start_update.elapsed_time(end_update) / 1000.0


    def predict(self, motion_vectors, frame_type, frame_shape):
        if self.measure_timing:
            start_predict = torch.cuda.Event(enable_timing=True)
            end_predict = torch.cuda.Event(enable_timing=True)
            start_predict.record()

        # if there are no boxes skip prediction step
        if np.shape(self.boxes)[0] == 0:
            return

        # I frame has no motion vectors
        if frame_type != "I":
            motion_vectors = self._preprocess_motion_vectors(motion_vectors, (frame_shape[1], frame_shape[0]))
            # motion vectors is now a list of tensors where the first item is the P vectors and the second item the B vectors

            sample = self.standardize_motion_vectors({"motion_vectors": motion_vectors})
            motion_vectors = sample["motion_vectors"]

            for i in range(len(motion_vectors)):
                # swap channel order of motion vectors from BGR to RGB
                motion_vectors[i] = motion_vectors[i][..., [2, 1, 0]]
                # swap motion vector axes so that shape is (B, C, H, W) instead of (B, H, W, C)
                motion_vectors[i] = motion_vectors[i].permute(0, 3, 1, 2)

            self.last_motion_vectors = motion_vectors

        # pre process boxes
        boxes_prev = np.copy(self.boxes)
        boxes_prev = torch.from_numpy(boxes_prev)
        num_boxes = (boxes_prev.shape)[0]
        # model expects boxes in format [frame_idx, xmin, ymin, w, h]
        boxes_prev_tmp = torch.zeros(num_boxes, 5).float()
        boxes_prev_tmp[:, 1:] = boxes_prev
        boxes_prev = boxes_prev_tmp
        boxes_prev = boxes_prev.unsqueeze(0)  # add batch dimension

        boxes_prev_ = boxes_prev.clone()
        if self.mvs_mode == "dense":
            boxes_prev_[:, 1:] = boxes_prev_[:, 1:] / 16.0
        boxes_prev_ = boxes_prev_.to(self.device)

        motion_vectors_p = self.last_motion_vectors[0]
        motion_vectors_p = motion_vectors_p.to(self.device)
        try:
            motion_vectors_b = self.last_motion_vectors[1]
        except IndexError:
            motion_vectors_b = None
        else:
            motion_vectors_b = motion_vectors_b.to(self.device)

        # feed into model, retrieve output
        with torch.set_grad_enabled(False):
            if self.measure_timing:
                start_inference = torch.cuda.Event(enable_timing=True)
                end_inference = torch.cuda.Event(enable_timing=True)
                start_inference.record()
            if self.mvs_mode == "upsampled":
                velocities_pred = self.model(motion_vectors_p, boxes_prev_)
            elif self.mvs_mode == "dense":
                velocities_pred = self.model(motion_vectors_p, motion_vectors_b,
                    boxes_prev_)
            if self.measure_timing:
                end_inference.record()
                torch.cuda.synchronize()
                self.last_inference_dt = start_inference.elapsed_time(end_inference) / 1000.0

            # make sure output is on CPU
            velocities_pred = velocities_pred.cpu()
            if self.mvs_mode == "upsampled":
                velocities_pred = velocities_pred.view(1, -1, 4)
            elif self.mvs_mode == "dense":
                velocities_pred = velocities_pred.view(1, -1, 2)
            velocities_pred = velocities_pred[0, ...]

            # undo the standardization of predicted velocities
            sample = self.standardize_velocities({"velocities": velocities_pred})
            velocities_pred = sample["velocities"]

        # compute boxes from predicted velocities
        boxes_prev = boxes_prev[0, ...]
        boxes_prev = boxes_prev[..., 1:5]
        if self.mvs_mode == "upsampled":
            self.boxes = box_from_velocities(boxes_prev, velocities_pred).numpy()
        elif self.mvs_mode == "dense":
            self.boxes = box_from_velocities_2d(boxes_prev, velocities_pred).numpy()

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
