import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from coviar import load

from lib.model.tracknet import TrackNet
from lib.utils import count_params, weight_checksum
from lib.dataset.dataset import TrackDataset


num_epochs = 120
batch_size = 2
seq_len = 3

datasets = {x: TrackDataset(root_dir='data', mode=x, seq_length=seq_len) for x in ["train", "val"]}

print("Dataset stats:")
for mode, dataset in datasets.items():
    print("{} dataset has {} samples".format(mode, len(dataset)))

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
    shuffle=False, num_workers=0) for x in ["train", "val"]}

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: ", device)

tracknet = TrackNet()
tracknet = tracknet.to(device)
tracknet.device = device

criterion = nn.SmoothL1Loss(reduction="sum")
optimizer = optim.Adam(tracknet.parameters(), lr=0.01)

print("TrackNet Parameter Count:", count_params(tracknet))


for epoch in range(num_epochs):

    for phase in ["train", "val"]:
        if phase == "train":
            tracknet.train()
        else:
            tracknet.eval()

        for step, sample in enumerate(dataloaders[phase]):

            #frames = sample["frames"]
            mvs_residuals = sample["mvs_residuals"]
            velocities = sample["velocities"]
            boxes_prev = sample["boxes_prev"]
            #boxes = sample["boxes"]
            num_boxes = sample["num_boxes"]

            # change format of mvs_residuals
            mvs_residuals = mvs_residuals.permute(0, 1, 4, 2, 3)  # change to [batch, seq_len, C, H, W]

            # insert batch index into boxes_prev
            boxes_prev_tmp = torch.zeros((*boxes_prev.shape[:-1], 5))
            boxes_prev_tmp[..., 1:] = boxes_prev
            for batch_index in range(batch_size):
                boxes_prev_tmp[batch_index, ..., 0] = batch_index
            boxes_prev = boxes_prev_tmp

            # change box format to [x1, x2, y1, y2]
            boxes_prev[..., -2] = boxes_prev[..., -2] + boxes_prev[..., -4]
            boxes_prev[..., -1] = boxes_prev[..., -1] + boxes_prev[..., -3]

            # pick out gt velocitiy for last time step
            velocities = velocities[:, -1, :, :]

            #print(boxes_prev)

            mvs_residuals = mvs_residuals.to(device)
            boxes_prev = boxes_prev.to(device)
            velocities = velocities.to(device)

            tracknet.zero_grad()
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                velocities_pred = tracknet(mvs_residuals, boxes_prev)

                #print(velocities_pred.shape)
                #print(velocities.shape)

                loss = criterion(velocities_pred, velocities)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            print(phase, "epoch", epoch, "step", step, "weight_checksum = ", weight_checksum(tracknet), "loss = ", loss.item())
