import os
import pickle
from tqdm import tqdm
from video_cap import VideoCap
from seqinfo import scales, frame_rates, dir_names


CODEC = "h264"  # "h264" or "mpeg4"
DATASET = "MOT16"

if __name__ == "__main__":

    for mode in ["train", "test"]:
        for dir_name, frame_rate in zip(dir_names[DATASET][mode], frame_rates[DATASET][mode]):
            current_path = os.path.join(DATASET, mode, dir_name)
            for scale in scales:

                output_path = os.path.join(current_path, "mvs-{}-{}".format(CODEC, scale))
                os.makedirs(output_path, exist_ok=True)

                video_file = os.path.join(current_path, "{}-{}-{}.mp4".format(dir_name, CODEC, scale))

                cap = VideoCap()
                ret = cap.open(video_file)
                if not ret:
                    raise RuntimeError("Could not open the video file")

                print("Processing video {}".format(video_file))

                frame_idx = 1

                while True:
                    ret, frame, motion_vectors, frame_type, _ = cap.read()
                    if not ret:
                        break

                    data_item = {
                        "motion_vectors": motion_vectors,
                        "frame_type": frame_type,
                    }

                    pickle.dump(data_item, open(os.path.join(output_path, "{:06d}.pkl".format(frame_idx)), "wb"))
                    frame_idx += 1

                cap.release()
