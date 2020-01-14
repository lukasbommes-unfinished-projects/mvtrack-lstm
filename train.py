import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from coviar import load

from lib.model.tracknet import TrackNet
from lib.utils import count_params, weight_checksum


# train LSTM on single sample
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

tracknet = TrackNet()
tracknet = tracknet.to(device)
tracknet.device = device

print("TrackNet Parameter Count:", count_params(tracknet))

batch_size = 2
seq_len = 13
num_boxes = 6
mvs_residuals = torch.randn(batch_size, seq_len, 5, 563, 1000)  # [batch_size, seq_len, num_features]
boxes_prev = torch.randn(batch_size, seq_len, num_boxes, 5)  # [batch_size, seq_len, num_boxes, 5]
# set batch index for boxes_prev
boxes_prev[0, :, :, 0] = 0
boxes_prev[1, :, :, 0] = 1
print(boxes_prev)

velocities_gt = torch.randn(batch_size, seq_len, num_boxes, 4)

criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(tracknet.parameters(), lr=0.1)

print("boxes_prev shape", boxes_prev.shape)

# rescale mvs_residuals to be of shape [5, 540, 960] (also rescale values of mvs according to scale in x and y direction)

mvs_residuals = mvs_residuals.to(device)
boxes_prev = boxes_prev.to(device)
velocities_gt = velocities_gt.to(device)

tracknet = tracknet.train()

for i in range(500):

    print("step = ", i)

    tracknet.zero_grad()

    velocities_pred = tracknet(mvs_residuals, boxes_prev)

    # TODO: make LSTM stateful, that is store hidden and cell states between subsequent samples

    loss = criterion(velocities_pred, velocities_gt)
    loss.backward()
    optimizer.step()

    print("weight_checksum = ", weight_checksum(tracknet))
    print("loss = ", loss)



print("###")
print("velocities_pred shape", velocities_pred.shape)

# mvs_residuals = []
# for frame_id in range(batch_size):
#     frame_data = pickle.load(open(os.path.join("mvs", "{:06d}.pkl".format(frame_id)), "rb"))
#     frame_data = torch.from_numpy(frame_data).unsqueeze(0)
#     mvs_residuals.append(frame_data)
#     print(frame_data.shape)
# mvs_residuals = torch.cat(mvs_residuals, axis=0)
#
# print(mvs_residuals.shape)

#num_epochs = 10
#for epoch in num_epochs:

#    velocities_pred = tracknet(mvs_residuals, boxes_prev)
