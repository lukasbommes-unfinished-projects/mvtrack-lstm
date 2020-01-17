import os
import glob
import torch
import numpy as np
import cv2
import coviar

from lib.visu import draw_boxes
from lib.dataset.loaders import load_detections
from lib.dataset.utils import compute_scaling_factor
from lib.tracking.tracker_OTCD import TrackerOTCD


if __name__ == "__main__":

    # reminder: when evaluating h264 models, use h264 videos
    #video_file = "data/MOT17/train/MOT17-02-FRCNN/MOT17-02-FRCNN-mpeg4-1.0.mp4"  # train set, static cam
    #video_file = "data/MOT17/train/MOT17-11-FRCNN/MOT17-11-FRCNN-mpeg4-1.0.mp4"  # train set, moving cam
    #video_file = "data/MOT17/test/MOT17-08-FRCNN/MOT17-08-FRCNN-mpeg4-1.0.mp4"  # test set, static cam
    #video_file = "data/MOT17/test/MOT17-12-FRCNN/MOT17-12-FRCNN-mpeg4-1.0.mp4"  # test set, moving cam
    video_file = "data/MOT17/train/MOT17-09-FRCNN/MOT17-09-FRCNN-mpeg4-1.0.mp4"  # val set, static cam
    #video_file = "data/MOT17/train/MOT17-10-FRCNN/MOT17-10-FRCNN-mpeg4-1.0.mp4"  # val set, moving cam

    detections_file = "data/MOT17/train/MOT17-09-FRCNN/det/det.txt"

    detector_interval = 20
    tracker_iou_thres = 0.1
    det_conf_threshold = 0.5
    state_thresholds = (0, 1, 10)
    gop_size = 12

    tracker_baseline = TrackerOTCD(
        iou_threshold=tracker_iou_thres,
        det_conf_threshold=det_conf_threshold,
        state_thresholds=state_thresholds,
        device=torch.device("cuda:0"),
        use_numeric_ids=True,
        measure_timing=True)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 640, 360)

    # load detections
    base_dir = str.split(video_file, "/")[:-1]
    num_frames = len(glob.glob(os.path.join(*base_dir, 'img1', '*.jpg')))
    det_boxes_all, det_scores_all = load_detections(detections_file, num_frames)

    frame_idx = 0
    step_wise = True

    # box colors
    color_detection = (0, 0, 150)
    color_tracker_baseline = (0, 0, 255)
    color_previous_baseline = (150, 150, 255)
    #color_tracker_deep = (0, 255, 255)
    #color_previous_deep = (150, 255, 255)

    prev_boxes_baseline = None
    #prev_boxes_deep = None

    while True:

        # load frame
        frame_file = os.path.join(*base_dir, "img1", "{:06d}.jpg".format(frame_idx + 1))
        frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)

        # load mvs_residuals
        gop_idx = int(frame_idx / gop_size)  # GOP starts from 0, while frame_id here starts from 1.
        in_group_idx = int(frame_idx % gop_size)  # the index in the group
        mv = coviar.load(video_file, gop_idx, in_group_idx, 1, False)
        residual = coviar.load(video_file, gop_idx, in_group_idx, 2, False)
        mvs_residuals = np.zeros((residual.shape[0], residual.shape[1], 5))
        mvs_residuals[:, :, 0:2] = mv  # XY
        mvs_residuals[:, :, 2:5] = residual  # BGR

        # scale frame and residuals to max size
        scaling_needed, scaling_factor = compute_scaling_factor(mvs_residuals)
        # scale mvs_residuals
        if scaling_needed:
            frame = cv2.resize(frame, None, None, fx=scaling_factor,
                fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
            mvs_residuals = cv2.resize(mvs_residuals, None, None, fx=scaling_factor,
                fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
            mvs_residuals[:, :, 0:2] = mvs_residuals[:, :, 0:2] * scaling_factor

        # draw color legend
        frame = cv2.putText(frame, "Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Baseline Previous Prediction", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous_baseline, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Baseline Tracker Prediction", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker_baseline, 2, cv2.LINE_AA)
        #frame = cv2.putText(frame, "Deep Previous Prediction", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous_deep, 2, cv2.LINE_AA)
        #frame = cv2.putText(frame, "Deep Tracker Prediction", (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker_deep, 2, cv2.LINE_AA)

        # update with detections
        if frame_idx % detector_interval == 0:
            det_boxes = det_boxes_all[frame_idx] * scaling_factor
            det_scores = det_scores_all[frame_idx] * scaling_factor

            tracker_baseline.update(mvs_residuals, det_boxes, det_scores)
            #tracker_deep.update(motion_vectors, frame_type, det_boxes, det_scores)
            if prev_boxes_baseline is not None:
               frame = draw_boxes(frame, prev_boxes_baseline, color=color_previous_baseline)
            prev_boxes_baseline = np.copy(det_boxes)
            # if prev_boxes_deep is not None:
            #    frame = draw_boxes(frame, prev_boxes_deep, color=color_previous_deep)
            # prev_boxes_deep = np.copy(det_boxes)

        # prediction by tracker
        else:
            tracker_baseline.predict(mvs_residuals)
            track_boxes_baseline = tracker_baseline.get_boxes()
            box_ids_baseline = tracker_baseline.get_box_ids()

            # tracker_deep.predict(motion_vectors, frame_type)
            # track_boxes_deep = tracker_deep.get_boxes()
            # box_ids_deep = tracker_deep.get_box_ids()

            if prev_boxes_baseline is not None:
               frame = draw_boxes(frame, prev_boxes_baseline, color=color_previous_baseline)
            prev_boxes_baseline = np.copy(track_boxes_baseline)

            # if prev_boxes_deep is not None:
            #    frame = draw_boxes(frame, prev_boxes_deep, color=color_previous_deep)
            # prev_boxes_deep = np.copy(track_boxes_deep)

            print(type(track_boxes_baseline))
            print(track_boxes_baseline)
            frame = draw_boxes(frame, track_boxes_baseline, box_ids=box_ids_baseline, color=color_tracker_baseline)
            #frame = draw_boxes(frame, track_boxes_deep, box_ids=box_ids_deep, color=color_tracker_deep)

        frame = draw_boxes(frame, det_boxes, scores=det_scores, color=color_detection)

        # print FPS
        print("### FPS ###")
        print("Baseline: Predict {}, Update {}".format(
            1/tracker_baseline.last_predict_dt, 1/tracker_baseline.last_update_dt))
        # print("Deep: Predict {}, Update {}, Inference {}".format(
        #     1/tracker_deep.last_predict_dt, 1/tracker_deep.last_update_dt,
        #     1/tracker_deep.last_inference_dt))

        frame_idx += 1
        cv2.imshow("frame", frame)

        # handle key presses
        # 'q' - Quit the running program
        # 's' - enter stepwise mode
        # 'a' - exit stepwise mode
        key = cv2.waitKey(1)
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

    cv2.destroyAllWindows()
