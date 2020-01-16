import numpy as np
#import cv2
import scipy
from scipy import optimize


def compute_iou(boxA, boxB):
    """Computes the Intersection over Union (IoU) for two bounding boxes.

    Args:
        boxA, boxB (`numpy.ndarray`): Bounding boxes [xmin, ymin, width, height]
            as arrays with shape (4,) and dtype float.

    Returns:
        IoU (`float`): The IoU of the two boxes. It is within the range [0, 1],
        0 meaning no overlap and 1 meaning full overlap of the two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs(boxA[2] * boxA[3])
    boxBArea = abs(boxB[2] * boxB[3])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def match_bounding_boxes(t_boxes, d_boxes, iou_threshold):
    """Matches detection boxes with tracked boxes based on IoU.

    This function can be used to find matches between sets of bounding boxes
    found with an object detector and tracked with a tracker. It yields three
    arrays with indices indicating which detected box corresponds to which
    tracked box. This information is needed to update the state of the tracker
    boxes with the correct detection box.

    Matching is performed by the Hungarian Algorithm applied to a cost matrix of
    IoUs of all possible pairs of boxes.

    Args:
        t_boxes (`numpy.ndarray`): Array of shape (T, 4) and dtype float of the
            T tracked bounding boxes in the format [xmin, ymin, width, height]
            each.

        d_boxes (`numpy.ndarray`): Array of shape (D, 4) and dtype float of the
            D detected bounding boxes in the format [xmin, ymin, width, height]
            each.

        iou_threshold (`float`): If the IoU of a detected and a tracked box
            exceeds this threshold they are considered as a match.

    Returns:
        matches (`numpy.ndarray`): Array of shape (M, 2) containing the indices
            of all M matched pairs of detection and tracking boxes. Each row in
            this array has the form [d, t] indicating that the `d`th detection
            box has been matched with the `t`th tracking box (d and t being the
            row indices of d_boxes and t_boxes).

        unmatched_trackers (`numpy.ndarray`): Array of shape (L,) containing
            the L row indices of all tracked boxes which could not be matched to
            any detected box (that is their IoU did not exceed the
            `iou_threshold`). This indicates an event, such as a previously
            tracked target leaving the scene.

        unmatched_detectors (`numpy.ndarray`): Array of shape (K,) containing
            the K row indices of all detected boxes which could not be matched
            to any tracked box. This indicates an event such as a new target
            entering the scene.
    """
    matches = []
    unmatched_trackers = []
    unmatched_detectors = []

    # compute IoU matrix for all possible matches of tracking and detection boxes
    iou_matrix = np.zeros([len(d_boxes), len(t_boxes)])
    for d, d_box in enumerate(d_boxes):
        for t, t_box in enumerate(t_boxes):
            iou_matrix[d, t] = compute_iou(d_box, t_box)
    # find matches between detection and tracking boxes that lead to maximum total IoU
    d_idx, t_idx = scipy.optimize.linear_sum_assignment(-iou_matrix)
    # find all detection boxes, which have no tracker yet
    unmatched_detectors = []
    for d in range(len(d_boxes)):
        if d not in d_idx:
            unmatched_detectors.append(d)
    # find all tracker boxes, which have not been detected anymore
    unmatched_trackers = []
    for t in range(len(t_boxes)):
        if t not in t_idx:
            unmatched_trackers.append(t)
    # filter out matches with low Iou
    matches = []
    for d, t in zip(d_idx, t_idx):
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detectors.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append([d, t])
    if len(matches) == 0:
        matches = np.empty((0, 2))
    else:
        matches = np.vstack(matches)

    # sort descending for later deletion
    unmatched_trackers = np.array(sorted(unmatched_trackers, reverse=True))
    unmatched_detectors = np.array(sorted(unmatched_detectors, reverse=True))
    return matches, unmatched_trackers, unmatched_detectors
