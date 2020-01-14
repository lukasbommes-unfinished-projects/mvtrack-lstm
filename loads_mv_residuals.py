from coviar import load
import pickle
import os
import numpy as np


sequence = "MOT17-09-FRCNN-mpeg4-1.0.mp4"
out_dir = "mvs"

step_wise = True
frame_id = 0
gop_size = 10

for frame_id in range(10): #525):
    print(frame_id)

    gop_idx = int((frame_id - 1) / gop_size)  # GOP starts from 0, while frame_id here starts from 1.
    in_group_idx = int((frame_id - 1) % gop_size)  # the index in the group

    mv = load(sequence, gop_idx, in_group_idx, 1, False)
    residual = load(sequence, gop_idx, in_group_idx, 2, False)

    print("mv shape", mv.shape)
    print("residual shape", residual.shape)

    frame_data = np.zeros((residual.shape[0], residual.shape[1], 5))
    frame_data[:, :, 0:2] = mv  # XY
    frame_data[:, :, 2:5] = residual  # BGR

    out_file = os.path.join(out_dir, "{:06d}.pkl".format(frame_id))
    pickle.dump(frame_data, open(out_file, "wb"))
