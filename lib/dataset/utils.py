import numpy as np


def compute_scaling_factor(mvs_residuals, max_scale=1000):
    current_scale = np.max(mvs_residuals.shape[:2])
    scaling_needed = False
    scaling_factor = 1
    if current_scale > max_scale:
        scaling_needed = True
        scaling_factor = max_scale / current_scale
    return scaling_needed, scaling_factor


def convert_to_tlbr(boxes):
    """[xmin, ymin, width, height] -> [x1, y1, x2, y2]"""
    boxes_ = boxes.clone()
    boxes_[..., -2] = boxes_[..., -2] + boxes_[..., -4] - 1
    boxes_[..., -1] = boxes_[..., -1] + boxes_[..., -3] - 1
    return boxes_


def convert_to_tlwh(boxes):
    """[x1, y1, x2, y2] -> [xmin, ymin, width, height]"""
    boxes_ = boxes.clone()
    boxes_[..., -2] = boxes_[..., -2] - boxes_[..., -4] + 1
    boxes_[..., -1] = boxes_[..., -1] - boxes_[..., -3] + 1
    return boxes_
