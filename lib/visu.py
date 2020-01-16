import cv2
import numpy as np


def draw_boxes_on_motion_vector_image(mvs_image, bounding_boxes, color=(255, 255, 255)):
    for box in bounding_boxes:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[0] + box[2])
        ymax = int(box[1] + box[3])
        mvs_image = cv2.rectangle(mvs_image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_4)
    return mvs_image


def draw_motion_vectors(frame, motion_vectors, format='torch'):
    """Draw motion vectors onto the frame.

    Format can be either 'numpy' for motion vectors as returned directly by
    video_cap or 'torch' for motion vectors trasformed to a torch tensor.
    """
    if format == 'torch':
        if motion_vectors.shape[0] > 0:
            for x in range(motion_vectors.shape[2]):
                for y in range(motion_vectors.shape[1]):
                    motion_x = motion_vectors[0, y, x]
                    motion_y = motion_vectors[1, y, x]
                    start_pt = (x * 16 + 8, y * 16 + 8)  # the x,y coords correspond to the vector destination
                    end_pt = (start_pt[0] - motion_x, start_pt[1] - motion_y)  # vector source
                    frame = cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.3)
    elif format == 'numpy':
        if np.shape(motion_vectors)[0] > 0:
            num_mvs = np.shape(motion_vectors)[0]
            for mv in np.split(motion_vectors, num_mvs):
                start_pt = (mv[0, 5], mv[0, 6])
                end_pt = (mv[0, 3], mv[0, 4])
                if mv[0, 0] < 0:
                    frame = cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.3)
                else:
                    frame = cv2.arrowedLine(frame, start_pt, end_pt, (0, 255, 0), 1, cv2.LINE_AA, 0, 0.3)
    return frame


def draw_macroblocks(frame, motion_vectors, alpha=1.0):
    if np.shape(motion_vectors)[0] > 0:
        frame_cpy = frame.copy()
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            mb_xmin = int(mv[0, 5] - 0.5*mv[0, 1])  # x_dst - 1/2 m_w
            mb_ymin = int(mv[0, 6] - 0.5*mv[0, 2])  # y_dst - 1/2 m_h
            mb_xmax = int(mv[0, 5] + 0.5*mv[0, 1] - 1)  # x_dst + 1/2 m_w - 1
            mb_ymax = int(mv[0, 6] + 0.5*mv[0, 2] - 1)  # y_dst + 1/2 m_h - 1
            frame = cv2.rectangle(frame, (mb_xmin, mb_ymin), (mb_xmax, mb_ymax), (255, 255, 255))
        frame_overlay = cv2.addWeighted(frame, alpha, frame_cpy, 1-alpha, gamma=0)
    return frame_overlay


def draw_boxes(frame, bounding_boxes, box_ids=None, scores=None, color=(0, 255, 0)):
    for i, box in enumerate(bounding_boxes):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[0] + box[2])
        ymax = int(box[1] + box[3])
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_4)
        if box_ids is not None:
            frame = cv2.putText(frame, '{}'.format(str(box_ids[i])[:6]), (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        if scores is not None:
            frame = cv2.putText(frame, '{:.4f}'.format(scores[i]), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame


def draw_velocities(frame, bounding_boxes, velocities, scale=100):
    for box, velocity in zip(bounding_boxes, velocities):
        start_pt = (int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3]))
        end_pt = (int(start_pt[0] + scale*velocity[0]), int(start_pt[1] + scale*velocity[1]))
        frame = cv2.arrowedLine(frame, start_pt, end_pt, (255, 255, 255), 1, cv2.LINE_AA, 0, 0.1)
    return frame
