import pickle
import numpy as np


phase = "train"
step = 0

sample = pickle.load(open("sample_{}_{:06d}.pkl".format(phase, step), "rb"))

mvs_residuals = sample["mvs_residuals"]
velocities = sample["velocities"]
boxes_prev = sample["boxes_prev"]
boxes = sample["boxes"]
num_boxes_mask = sample["num_boxes_mask"]

print(mvs_residuals.shape)
print(velocities.shape)
print(boxes_prev.shape)
print(boxes.shape)
print(num_boxes_mask.shape)

print("boxes_prev", boxes_prev)
print("boxes", boxes)
print("velocities", velocities)

box = [698.9583, 217.1875,  87.5000, 197.9167]
box_prev = [696.8750, 217.7083,  86.9792, 197.3958]

velocity = 
