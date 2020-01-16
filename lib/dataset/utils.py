import numpy as np


def compute_scaling_factor(mvs_residuals, max_scale=1000):
    current_scale = np.max(mvs_residuals.shape[:2])
    scaling_needed = False
    scaling_factor = 1
    if current_scale > max_scale:
        scaling_needed = True
        scaling_factor = max_scale / current_scale
    return scaling_needed, scaling_factor
