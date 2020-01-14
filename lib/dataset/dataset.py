import os
import pickle
import glob
import torch
import torchvision
import cv2
import numpy as np

import coviar

from lib.dataset.loaders import load_groundtruth
from lib.dataset.velocities import velocities_from_boxes
from lib.visu import draw_boxes, draw_velocities, draw_motion_vectors


class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="data", mode="train", seq_length=3,
        max_scale=1000, gop_size=12, max_num_boxes=55, visu=True):

        self.sequences = {
            "train": [
                "MOT17/train/MOT17-02-FRCNN",  # static cam
                "MOT17/train/MOT17-04-FRCNN",  # static cam
                "MOT17/train/MOT17-05-FRCNN",  # moving cam
                "MOT17/train/MOT17-11-FRCNN",  # moving cam
                "MOT17/train/MOT17-13-FRCNN",  # moving cam
                "MOT15/train/ETH-Bahnhof",  # moving cam
                "MOT15/train/ETH-Sunnyday",  # moving cam
                "MOT15/train/KITTI-13",  # moving cam
                "MOT15/train/KITTI-17",  # static cam
                "MOT15/train/PETS09-S2L1",  # static cam
                "MOT15/train/TUD-Campus",  # static cam
                "MOT15/train/TUD-Stadtmitte"  # static cam
            ],
            "val": [
                "MOT17/train/MOT17-09-FRCNN",  # static cam
                "MOT17/train/MOT17-10-FRCNN"  # moving cam
            ]
        }

        self.root_dir = root_dir
        self.mode = mode
        self.seq_length = seq_length
        self.max_scale = max_scale
        self.gop_size = gop_size
        self.max_num_boxes = max_num_boxes
        self.visu = visu
        self.DEBUG = True

        self.index = []   # stores (sequence_idx, scale_idx, frame_idx) for available samples

        self.get_sequence_lengths_()
        self.load_groundtruth_()
        if self.DEBUG:
            print("Loaded ground truth files.")
        self.build_index_()
        if self.DEBUG:
            print("Built dataset index.")


    def get_sequence_lengths_(self):
        """Determine number of frames in each video sequence."""
        self.lens = []
        for sequence in self.sequences[self.mode]:
            frame_files = glob.glob(os.path.join(self.root_dir, sequence, "img1/*.jpg"))
            self.lens.append(len(frame_files))


    def load_groundtruth_(self):
        """Load ground truth boxes and IDs from annotation files."""
        self.gt_ids = []
        self.gt_boxes = []
        for sequence, num_frames in zip(self.sequences[self.mode], self.lens):
            gt_file = os.path.join(self.root_dir, sequence, "gt/gt.txt")
            gt_ids, gt_boxes, _ = load_groundtruth(gt_file, num_frames, only_eval=True)
            self.gt_ids.append(gt_ids)
            self.gt_boxes.append(gt_boxes)


    def load_frame_(self, sequence_idx, frame_idx):
        """Load, scale and return a single video frame."""
        frame_file = os.path.join(self.root_dir,
            self.sequences[self.mode][sequence_idx], "img1",
            "{:06d}.jpg".format(frame_idx + 1))
        frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
        return frame


    def build_index_(self):
        """Generate index of all usable frames of all sequences.

        Usable frames are those which have a ground truth annotation and for
        which the previous frame also has a ground truth annotation. Only those
        ground truth annotation for which the eval flag is set to 1 are considered
        (see `only_eval` parameter in load_groundtruth). If `excluse_keyframes`
        is True keyframes (frame type "I") are also excluded from the index.

        The index has the format [(0, 0, 2), (0, 0, 3), ..., (2, 2, 2),
        (2, 2, 3), (2, 2, 4)] where the first item of the tuple is the sequence
        index (0-based), the second item is the scale index (0-based) and the
        third item is the frame index (0-based) within this sequence.
        """
        # first lookup which items have a predessor item
        # note: frame_idx here is 0-based, while it is 1-based in the gt.txt file
        index_tmp = []
        for sequence_idx, sequence in enumerate(self.sequences[self.mode]):
            last_none = True
            for frame_idx in range(len(self.gt_ids[sequence_idx])):
                gt_ids = self.gt_ids[sequence_idx][frame_idx]
                gt_ids_prev = self.gt_ids[sequence_idx][frame_idx - 1]
                # exclude frames without gt annotation from index
                if gt_ids is None:
                    last_none = True
                    continue
                if last_none:
                    last_none = False
                    continue
                # check if ids can be matched, otherwise exclude frame
                _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev,
                    assume_unique=True, return_indices=True)
                if len(idx_1) == 0 and len(idx_0) == 0:
                    continue
                index_tmp.append((sequence_idx, frame_idx))
        # now check how we can arrange the pairs in a sequence
        for idx in range(len(index_tmp)):
            seq_idx_is_same = False
            frame_idx_is_incremental = False
            try:
                subseq = index_tmp[idx:idx+self.seq_length]
                # check if all sequence_idx in the subsequence are same
                if len(set([s[0] for s in subseq])) == 1:
                    seq_idx_is_same = True
                # check if frame_idx in the subsequence are incremental
                if set([s2[1]-s1[1] for s1, s2 in zip(subseq, subseq[1:]+subseq[:1])][:-1]) == {1}:
                    frame_idx_is_incremental = True
                if seq_idx_is_same and frame_idx_is_incremental and len(subseq) == self.seq_length:
                    self.index.append([subseq[0][0], [s[1] for s in subseq]])
            except IndexError:
                continue


    def compute_scaling_factor_(self, mvs_residuals):
        current_scale = np.max(mvs_residuals.shape[:2])
        scaling_needed = False
        if current_scale > self.max_scale:
            scaling_needed = True
            scaling_factor = self.max_scale / current_scale
        return scaling_needed, scaling_factor


    def __len__(self):
        """Return the total length of the dataset."""
        total_len = len(self.index)
        return total_len


    def __getitem__(self, idx):
        """Retrieve item with index `idx` from the dataset."""
        sequence_idx, frame_indices = self.index[idx]

        frames_seq = torch.tensor([], dtype=torch.uint8)
        mvs_residuals_seq = torch.tensor([], dtype=torch.float)
        boxes_prev_seq = torch.tensor([], dtype=torch.float)
        boxes_seq = torch.tensor([], dtype=torch.float)
        velocities_seq = torch.tensor([], dtype=torch.float)
        num_boxes_seq = torch.tensor([], dtype=torch.long)

        for frame_idx in frame_indices:
            frame = self.load_frame_(sequence_idx, frame_idx)

            gop_idx = int(frame_idx / self.gop_size)  # GOP starts from 0, while frame_id here starts from 1.
            in_group_idx = int(frame_idx % self.gop_size)  # the index in the group

            sequence_root = os.path.join(self.root_dir, self.sequences[self.mode][sequence_idx])
            sequence_name = "{}-mpeg4-1.0.mp4".format(str.split(sequence_root,"/")[-1])
            sequence = os.path.join(sequence_root, sequence_name)

            mv = coviar.load(sequence, gop_idx, in_group_idx, 1, False)
            residual = coviar.load(sequence, gop_idx, in_group_idx, 2, False)

            mvs_residuals = np.zeros((residual.shape[0], residual.shape[1], 5))
            mvs_residuals[:, :, 0:2] = mv  # XY
            mvs_residuals[:, :, 2:5] = residual  # BGR

            # compute scaling factor so that longer side is at maximum equal max_scale
            scaling_needed, scaling_factor = self.compute_scaling_factor_(mvs_residuals)
            if scaling_needed:
                frame = cv2.resize(frame, None, None, fx=scaling_factor,
                    fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
                mvs_residuals = cv2.resize(mvs_residuals, None, None, fx=scaling_factor,
                    fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
                mvs_residuals[:, :, 0:2] = mvs_residuals[:, :, 0:2] * scaling_factor

            if self.visu:
                sequence_name = str.split(self.sequences[self.mode][sequence_idx], "/")[-1]
                cv2.putText(frame, 'Sequence: {}'.format(sequence_name), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Frame Idx: {}'.format(frame_idx), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # get ground truth boxes and update previous boxes and ids
            gt_boxes = self.gt_boxes[sequence_idx][frame_idx]
            gt_ids = self.gt_ids[sequence_idx][frame_idx]
            gt_boxes_prev = self.gt_boxes[sequence_idx][frame_idx - 1]
            gt_ids_prev = self.gt_ids[sequence_idx][frame_idx - 1]

            # scale boxes
            if scaling_needed:
                gt_boxes = gt_boxes*scaling_factor
                gt_boxes_prev = gt_boxes_prev*scaling_factor

            # match ids with previous ids and compute box velocities
            _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev, assume_unique=True, return_indices=True)
            boxes = torch.from_numpy(gt_boxes[idx_1]).float()
            boxes_prev = torch.from_numpy(gt_boxes_prev[idx_0]).float()

            velocities = velocities_from_boxes(boxes_prev, boxes)

            if self.visu:
                frame = draw_boxes(frame, boxes, gt_ids, color=(255, 255, 255))
                frame = draw_boxes(frame, boxes_prev, gt_ids_prev, color=(200, 200, 200))

            # pad boxes and velocities to the maximum number of ground truth boxes
            num_boxes = (boxes.shape)[0]
            boxes_prev_padded = torch.zeros(self.max_num_boxes, 4).float()
            boxes_prev_padded[:num_boxes, :] = boxes_prev
            # pad boxes prev
            boxes_prev = boxes_prev_padded
            boxes_padded = torch.zeros(self.max_num_boxes, 4).float()
            boxes_padded[:num_boxes, :] = boxes
            boxes = boxes_padded
            # pad velocites
            velocities_padded = torch.zeros(self.max_num_boxes, 4).float()
            velocities_padded[:num_boxes, :] = velocities
            velocities = velocities_padded

            frame = torch.from_numpy(frame).type(torch.uint8).unsqueeze(0)
            mvs_residuals = torch.from_numpy(mvs_residuals).type(torch.float).unsqueeze(0)
            velocities = velocities.float().unsqueeze(0)
            boxes_prev = boxes_prev.float().unsqueeze(0)
            boxes = boxes.float().unsqueeze(0)
            num_boxes = torch.tensor(num_boxes).unsqueeze(0)

            frames_seq = torch.cat((frames_seq, frame), axis=0)
            mvs_residuals_seq = torch.cat((mvs_residuals_seq, mvs_residuals), axis=0)
            velocities_seq = torch.cat((velocities_seq, velocities), axis=0)
            boxes_prev_seq = torch.cat((boxes_prev_seq, boxes_prev), axis=0)
            boxes_seq = torch.cat((boxes_seq, boxes), axis=0)
            num_boxes_seq = torch.cat((num_boxes_seq, num_boxes), axis=0)

        sample = {
            "frames": frames_seq,       # [seq_len, H, W, C=3], C order BGR
            "mvs_residuals": mvs_residuals_seq,  # [seq_len, H, W, C=5], C 0:2 mvs XY, C 2:5 residual BGR
            "velocities": velocities_seq,  # [seq_len, max_num_boxes, 4], row format [vxc, vyc, vw, vh]
            "boxes_prev": boxes_prev_seq,  # [seq_len, max_num_boxes, 4], row format [xmin, ymin, w, h]
            "boxes": boxes_seq,            # [seq_len, max_num_boxes, 4], row format [xmin, ymin, w, h]
            "num_boxes": num_boxes_seq     # [seq_len], number of valid boxes in each seq item, needed to revert padding
        }

        return sample


# run as python -m lib.dataset.dataset from root dir
if __name__ == "__main__":

    batch_size = 1
    seq_len = {"train": 3, "val": 3}

    datasets = {x: MotionVectorDataset(root_dir='data', mode=x, seq_length=seq_len[x]) for x in ["train", "val"]}

    print("Dataset stats:")
    for mode, dataset in datasets.items():
        print("{} dataset has {} samples".format(mode, len(dataset)))

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
        shuffle=False, num_workers=0) for x in ["train", "val"]}

    step_wise = True

    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len["train"]):
            cv2.namedWindow("frame-{}-{}".format(batch_idx, seq_idx), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame-{}-{}".format(batch_idx, seq_idx), 640, 360)
            cv2.namedWindow("motion_vectors-{}-{}".format(batch_idx, seq_idx), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_vectors-{}-{}".format(batch_idx, seq_idx), 640, 360)
            cv2.namedWindow("residuals-{}-{}".format(batch_idx, seq_idx), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("residuals-{}-{}".format(batch_idx, seq_idx), 640, 360)

    for step, sample in enumerate(dataloaders["train"]):

        print("step", step, "sample", sample["frames"].shape,
            sample["mvs_residuals"].shape, sample["velocities"].shape,
            sample["boxes"].shape, sample["boxes_prev"].shape, sample["num_boxes"].shape)


        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len["train"]):

                frame = sample["frames"][batch_idx, seq_idx, ...].numpy()

                mvs_residuals = sample["mvs_residuals"][batch_idx, seq_idx, ...].numpy().astype(np.int8)
                residuals = mvs_residuals[..., 2:5]

                motion_vectors = np.zeros(shape=(mvs_residuals.shape[0], int(mvs_residuals.shape[1]), 3))
                motion_vectors[..., 2] = mvs_residuals[..., 0]
                motion_vectors[..., 1] = mvs_residuals[..., 1]
                motion_vectors = motion_vectors.astype(np.int8)

                print("step: {}".format(step))

                cv2.imshow("frame-{}-{}".format(batch_idx, seq_idx), frame)
                cv2.imshow("motion_vectors-{}-{}".format(batch_idx, seq_idx), motion_vectors)
                cv2.imshow("residuals-{}-{}".format(batch_idx, seq_idx), residuals)

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
