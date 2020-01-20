import os
import pickle
import copy
import time
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from coviar import load

from lib.model.tracknet import TrackNet
from lib.loss import smooth_l1_loss
from lib.utils import count_params, weight_checksum, compute_mean_iou
from lib.dataset.dataset import TrackDataset
from lib.dataset.velocities import bbox_transform_inv_otcd, box_from_velocities
from lib.dataset.utils import convert_to_tlbr, convert_to_tlwh


torch.set_printoptions(precision=10)

# CURRENT ISSUES
# sigma factor 1.5 for OTCD T-CNN, use appropiate velocity <-> box conversion functions from OTCD (update below for mean IoU comutation)
# velocity_pred has to be denormlized with bbox_reg_mean and bbox_reg_std stats
# see if velocity needs to be normlized with stats before computing loss
# make sure sigma factor is set correctly

# EXPERIMENTS FOR PAPER:
# Check if scaling factor in tracknet is really 1/16. might be changed due to additional layers
# 1) Vary seq_len
# 2) Try if propagating the hidden and cell states between batches is helpful
# 3) Try if adding a second LSTM layer helps
# 4) Try if unlocking weights of some of the top layers (e.g. conv1x1) for training helps

num_epochs = 100
batch_size = 1
seq_len = 3
learning_rate = 0.01  # pink: 0.1, türkis: 0.01, , # orange: 0.001, blue-green: 0.01
weight_decay = 0.0001
scheduler_steps = [8, 16, 24]
scheduler_factor = 0.1
sigma = 1.5
gpu = 0

# for velocity normalization
bbox_reg_mean = torch.tensor([0.0, 0.0, 0.0, 0.0])
bbox_reg_std = torch.tensor([0.1, 0.1, 0.2, 0.2])

write_tensorboard_log = True
save_model = True
log_to_file = True
save_model_every_epoch = True

datasets = {x: TrackDataset(root_dir='data', mode=x, batch_size=batch_size,
    seq_length=seq_len) for x in ["train", "val"]}

# print("Dataset stats:")
# for mode, dataset in datasets.items():
#     print("{} dataset has {} samples".format(mode, len(dataset)))

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
    shuffle=False, num_workers=4) for x in ["train", "val"]}

device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

tracknet = TrackNet()
tracknet = tracknet.to(device)
tracknet.device = device

#criterion = nn.SmoothL1Loss(reduction="mean")
optimizer = optim.Adam(tracknet.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,
    gamma=scheduler_factor)
#scheduler = None

# create output directory
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = os.path.join("models", "tracker", date)
if log_to_file or save_model:
    os.makedirs(outdir, exist_ok=True)

if write_tensorboard_log:
    writer = SummaryWriter()

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
logger.addHandler(ch)
if log_to_file:
    fh = logging.FileHandler(os.path.join(outdir, 'train.log'))
    logger.addHandler(fh)

logger.info("Model will be trained with the following options")
logger.info(f"outdir: {outdir}")
logger.info(f"batch_size: {batch_size}")
logger.info(f"seq_len: {seq_len}")
logger.info(f"learning_rate: {learning_rate}")
logger.info(f"num_epochs: {num_epochs}")
logger.info(f"weight_decay: {weight_decay}")
logger.info(f"scheduler_steps: {scheduler_steps}")
logger.info(f"scheduler_factor: {scheduler_factor}")
logger.info(f"sigma: {sigma}")
logger.info(f"gpu: {gpu}")
logger.info(f"model: {tracknet}")
logger.info(f"model requires_grad: {[p.requires_grad for p in tracknet.parameters()]}")
logger.info("model param count: {} (of which trainable: {})".format(*count_params(tracknet)))
logger.info(f"optimizer: {optimizer}")

tstart = time.time()

best_loss = 99999.0
best_mean_iou = 0.0
iterations = {"train": 0, "val": 0}

logger.info("Weight sum before training: {}".format(weight_checksum(tracknet)))

for epoch in range(num_epochs):

    # get current learning rate
    learning_rate = 0
    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']

    logger.info("Epoch {}/{} - Learning rate: {}".format(epoch, num_epochs-1, learning_rate))
    if write_tensorboard_log:
        writer.add_scalar('Learning Rate', learning_rate, epoch)

    for phase in ["train", "val"]:
        if phase == "train":
            tracknet.train()
        else:
            tracknet.eval()

        running_loss = []
        running_mean_iou = []

        for step, sample in enumerate(dataloaders[phase]):

            mvs_residuals = sample["mvs_residuals"]
            velocities = sample["velocities"]
            boxes_prev = sample["boxes_prev"]
            boxes = sample["boxes"]
            num_boxes_mask = sample["num_boxes_mask"]

            # print(mvs_residuals.shape)
            # print(velocities.shape)
            # print(boxes_prev.shape)
            # print(boxes.shape)
            # print(num_boxes_mask.shape)

            # change format of mvs_residuals
            mvs_residuals = mvs_residuals.permute(0, 1, 4, 2, 3)  # change to [batch, seq_len, C, H, W]

            # insert batch index into boxes_prev
            boxes_prev_ = boxes_prev.clone()
            boxes_prev_tmp = torch.zeros((*boxes_prev_.shape[:-1], 5))
            boxes_prev_tmp[..., 1:] = boxes_prev_
            for batch_index in range(batch_size):
                boxes_prev_tmp[batch_index, ..., 0] = batch_index
            boxes_prev_ = boxes_prev_tmp

            # change box format to [x1, x2, y1, y2]
            boxes_prev_ = convert_to_tlbr(boxes_prev_)

            # pick out velocity for lasst timestep in sequence
            velocities = velocities[:, -1, :, :]

            # normalize velocities
            #print("before", velocities)
            velocities = ((velocities - bbox_reg_mean.expand_as(velocities))
                / bbox_reg_std.expand_as(velocities))
            #print("after", velocities)

            #print(boxes_prev)

            mvs_residuals = mvs_residuals.to(device)
            boxes_prev_ = boxes_prev_.to(device)
            velocities = velocities.to(device)

            tracknet.zero_grad()
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                #print("boxes_prev_ shape", boxes_prev_.shape)
                #print("mvs_residuals shape", mvs_residuals.shape)
                velocities_pred = tracknet(mvs_residuals, boxes_prev_)
                #print("velocities_pred shape", velocities_pred.shape)
                #print("num_boxes_mask", num_boxes_mask)

                # velocities_pred: [batch_size, -1, 4]
                inside_weight = velocities_pred.new(batch_size, velocities.size(1), 4).zero_()
                outside_weight = velocities_pred.new(batch_size, velocities.size(1), 4).zero_()
                for bs_idx in range(batch_size):
                    num_boxes = num_boxes_mask[bs_idx, -1].nonzero().shape[0]
                    if num_boxes > 0:
                        inside_weight[bs_idx, 0:num_boxes, :] = 1.0
                        outside_weight[bs_idx, 0:num_boxes, :] = 1.0 / num_boxes

                loss = smooth_l1_loss(velocities_pred, velocities, inside_weight, outside_weight, dim=[2, 1])
                loss = loss.mean()

                #loss = criterion(velocities_pred, velocities)
                #print("loss", loss.item())

                if phase == "train":
                    if write_tensorboard_log:
                        params_before_update = [p.detach().clone() for p in tracknet.parameters()]

                    loss.backward()
                    optimizer.step()

                    if write_tensorboard_log:
                        params_after_update = [p.detach().clone() for p in tracknet.parameters()]
                        params_norm = torch.norm(torch.cat([p.flatten() for p in params_before_update], axis=0))
                        updates = [(pa - pb).flatten() for pa, pb in zip(params_after_update, params_before_update)]
                        updates_norm = torch.norm(torch.cat(updates, axis=0))
                        writer.add_scalar('update to weight ratio', updates_norm / params_norm, iterations["train"])

            running_loss.append(loss.item())

            boxes = boxes.detach().cpu()
            boxes_prev = boxes_prev.detach().cpu()
            velocities_pred = velocities_pred.detach().cpu()

            # log loss and mean IoU of all predicted and ground truth boxes
            mean_iou = 0
            for batch_idx in range(batch_size):
                velocities_pred_tmp = velocities_pred[batch_idx, num_boxes_mask[batch_idx, -1, :], :]
                velocities_tmp = velocities[batch_idx, num_boxes_mask[batch_idx, -1, :], :]
                boxes_prev_tmp = boxes_prev[batch_idx, -1, num_boxes_mask[batch_idx, -1, :], :]
                boxes_prev_tmp = convert_to_tlbr(boxes_prev_tmp)
                boxes_prev_tmp = boxes_prev_tmp.unsqueeze(0)
                boxes_tmp = boxes[batch_idx, -1, num_boxes_mask[batch_idx, -1, :], :]
                velocities_pred_tmp = velocities_pred_tmp.view(-1, 4) * bbox_reg_std + bbox_reg_mean
                velocities_pred_tmp = velocities_pred_tmp.view(1, -1, 4)
                boxes_pred = bbox_transform_inv_otcd(boxes=boxes_prev_tmp, deltas=velocities_pred_tmp, sigma=sigma, add_one=False)#.squeeze().numpy()
                boxes_prev_tmp = boxes_prev_tmp[0, ...]
                boxes_pred = boxes_pred[0, ...]
                velocities_pred_tmp = velocities_pred_tmp[0, ...]
                boxes_pred = convert_to_tlwh(boxes_pred)
                if phase == "val":
                    print("### batch_idx: {}".format(batch_idx))
                    print("velocities", velocities_tmp[:8, :])
                    print("velocities_pred", velocities_pred_tmp[:8, :])
                    print("boxes_prev", boxes_prev_tmp[:8, :])
                    print("boxes", boxes_tmp[:8, :])
                    print("boxes_pred", boxes_pred[:8, :])
                mean_iou = mean_iou + compute_mean_iou(boxes_pred, boxes_tmp)
            mean_iou = mean_iou / batch_size
            running_mean_iou.append(mean_iou)

            print(phase, "epoch", epoch, "step", step, "weight sum = ", weight_checksum(tracknet), "loss = ", loss.item(), "mean_iou = ", mean_iou, "lr = ", learning_rate)
            if write_tensorboard_log:
                writer.add_scalar('Loss/{}'.format(phase), loss.item(), iterations[phase])
                writer.add_scalar('Mean IoU/{}'.format(phase), mean_iou, iterations[phase])

            iterations[phase] += 1

        # epoch loss and IoU
        epoch_loss = np.mean(running_loss)
        epoch_mean_iou = np.mean(running_mean_iou)
        logger.info('{} Loss: {}; {} Mean IoU: {}'.format(phase, epoch_loss, phase, epoch_mean_iou))
        if write_tensorboard_log:
            writer.add_scalar('Epoch Loss/{}'.format(phase), epoch_loss, epoch)
            writer.add_scalar('Epoch Mean IoU/{}'.format(phase), epoch_mean_iou, epoch)

        if phase == "val":
            if epoch_loss <= best_loss:
                best_loss = epoch_loss
                if save_model:
                    best_model_wts = copy.deepcopy(tracknet.state_dict())
                    logger.info("Saving model with lowest loss so far")
                    torch.save(best_model_wts, os.path.join(outdir, "model_lowest_loss.pth"))
            if epoch_mean_iou >= best_mean_iou:
                best_mean_iou = epoch_mean_iou
                if save_model:
                    best_model_wts = copy.deepcopy(tracknet.state_dict())
                    logger.info("Saving model with highest IoU so far")
                    torch.save(best_model_wts, os.path.join(outdir, "model_highest_iou.pth"))
            if save_model and save_model_every_epoch:
                best_model_wts = copy.deepcopy(tracknet.state_dict())
                torch.save(best_model_wts, os.path.join(outdir, "model_epoch_{}.pth".format(epoch)))

    if scheduler and epoch+1 in scheduler_steps:
        scheduler.step()

time_elapsed = time.time() - tstart
logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
logger.info('Lowest validation loss: {}'.format(best_loss))

logger.info("Weight sum after training: {}".format(weight_checksum(tracknet)))

if save_model:
    best_model_wts = copy.deepcopy(tracknet.state_dict())
    torch.save(best_model_wts, os.path.join(outdir, "model_final.pth"))

if write_tensorboard_log:
    writer.close()
