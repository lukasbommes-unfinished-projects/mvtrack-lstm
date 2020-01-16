import numpy as np
import cv2
from coviar import load


def scale_image(frame, short_side_min_len=600, long_side_max_len=1000):
    """Scale the input frame to match minimum and maximum size requirement."""
    # determine the scaling factor
    frame_size_min = np.min(frame.shape[0:2])
    frame_size_max = np.max(frame.shape[0:2])
    scaling_factor = float(short_side_min_len) / float(frame_size_min)
    if np.round(scaling_factor * frame_size_max) > long_side_max_len:
        scaling_factor = float(long_side_max_len) / float(frame_size_max)
    # scale the frame
    print(scaling_factor)
    frame = cv2.resize(frame, None, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    return frame, scaling_factor


#sequence = "output.mp4"  # this one works better as it uses the author's encoding settings
#sequence = "MOT17-09.avi"
#sequence = "output2.mp4"
#sequence = "out.mp4"
sequence = "MOT17-09-FRCNN-mpeg4-1.0.mp4"
#sequence = "ETH-Pedcross2-mpeg4-1.0.mp4"

step_wise = True
frame_id = 0
gop_size = 12

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 640, 360)
cv2.namedWindow("residuals", cv2.WINDOW_NORMAL)
cv2.resizeWindow("residuals", 640, 360)

for frame_id in range(145):

    print(frame_id)

    gop_idx = int((frame_id) / gop_size)  # GOP starts from 0, while frame_id  here starts from 1.
    in_group_idx = int((frame_id) % gop_size)  # the index in the group

    ret = load(sequence, gop_idx, in_group_idx, 1, False)
    #print(np.shape(ret))

    #print(np.min(ret), np.max(ret))
    #print(ret.dtype)

    image = np.zeros(shape=(ret.shape[0], int(ret.shape[1]), 3))#, fill_value=int(np.max(ret)), dtype=np.float32)
    image[:, :, 2] = ret[:, :, 0]  # red channel corresponds to x component of MVs
    image[:, :, 1] = ret[:, :, 1]  # green channel corresponds to y component of MVs
    image = image.astype(np.int8)

    residuals = load(sequence, gop_idx, in_group_idx, 2, False)
    residuals = residuals.astype(np.int8)

    cv2.imshow("frame", image)
    cv2.imshow("residuals", residuals)
    key = cv2.waitKey(25)
    if not step_wise and key == ord('s'):
        step_wise = True
    if key == ord('q'):
        break
    if step_wise:
        while True:
            key = cv2.waitKey(1)
            if key == ord('s'):
                break
            elif key == ord('a'):
                step_wise = False
                break
