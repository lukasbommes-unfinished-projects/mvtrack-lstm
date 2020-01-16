import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision

from lib.model.resnet_atrous import resnet18
from lib.model.convrnn import Conv2dLSTM
from lib.utils import load_pretrained_weights_to_modified_resnet, \
    load_pretrained_weights, count_params, weight_checksum


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.FIXED_BLOCKS = 1

        # load pretrained weights
        resnet = resnet18()
        resnet_weights = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        load_pretrained_weights_to_modified_resnet(resnet, resnet_weights)

        in_channels = 5  # mvs + residuals

        base = [
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.relu,
            resnet.layer2,
            resnet.relu,
            resnet.layer3,
            resnet.relu,
            resnet.layer4,
            resnet.relu]
        self.RCNN_base = nn.Sequential(*base)

        self.RCNN_bbox_base = nn.Conv2d(512, 4*7*7, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)

        assert (0 <= self.FIXED_BLOCKS <= 4) # set this value to 0, so we can train all blocks
        if self.FIXED_BLOCKS >= 4: # fix all blocks
            for p in self.RCNN_base[10].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 3: # fix first 3 blocks
            for p in self.RCNN_base[8].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 2: # fix first 2 blocks
            for p in self.RCNN_base[6].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 1: # fix first 1 block
            for p in self.RCNN_base[4].parameters(): p.requires_grad = False

    def forward(self, mvs_residuals):
        # motion vector are of shape [B, C, H, W] with batch size B
        # channels are in RGB order where red is x motion and green is y motion
        # boxes_prev are of shape [B, K, 5] with batch size B and number of boxes K
        # row format is [frame_idx, xmin, ymin, width, height]
        x = self.RCNN_base(mvs_residuals)
        return x


class TrackNet(nn.Module):
    def __init__(self, seq_length=3, pooling_size=7):
        super(TrackNet, self).__init__()

        self.DEBUG = False
        self.device = None
        self.pooling_size = pooling_size  # the ROIs are split into m x m regions
        self.base_scale = 1/16  # ration of base features to input size

        self.base = BaseNet()
        # load OTCD weigths into base net
        weights_file = "models/OTCD_trackingnet.pth"
        state_dict = torch.load(weights_file)
        self.base.load_state_dict(state_dict["model"])
        print("Loaded weights from {} into BaseNet.".format(weights_file))
        # fix base net weights
        for p in self.base.parameters(): p.requires_grad = False

        self.lstm = Conv2dLSTM(in_channels=512,
                               out_channels=196,
                               kernel_size=3,
                               num_layers=1,
                               bias=True,
                               batch_first=True,
                               dropout=0,
                               bidirectional=False,
                               stride=1,
                               dilation=1,
                               groups=1)

        self.pooling = nn.AvgPool2d(kernel_size=self.pooling_size, stride=self.pooling_size)
        self.conv1x1 = nn.Conv2d(196, 4*self.pooling_size*self.pooling_size, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.ps_roi_pool = torchvision.ops.PSRoIPool(output_size=(self.pooling_size, self.pooling_size), spatial_scale=self.base_scale)

        if self.DEBUG:
            print("base net param count: ", count_params(self.base))
            print("lstm param count: ", count_params(self.lstm))
            print("conv1x1 param count: ", count_params(self.conv1x1))


    def forward(self, mvs_residuals, boxes_prev):
        # mvs_residuals is a tensor of shape [batch_size, seq_len, 5, H, W] with C 0:2 MV XY and C 2:5 Residual BGR
        # boxes_prev is a tensor of shape [batch_size, seq_len, num_boxes, 5] with row format [frame_idx, x1, x2, y1, y2]
        if self.device is None:
            raise RuntimeError("Please set the device attribute explicitely.")

        batch_size, seq_len, C, H, W = mvs_residuals.shape
        num_boxes = boxes_prev.shape[-2]

        if self.DEBUG:
            print(mvs_residuals.shape)
            print(boxes_prev.shape)

        # apply CNN feature extractor to each element of the sequence
        out_tmp = torch.tensor([]).to(self.device)
        for i in range(seq_len):
            out_t = self.base(mvs_residuals[:, i, :, :, :])  # extract base features of shape [seq_len*batch_size, 196, ceil(H/16), ceil(W/16)]
            out_t  = out_t.unsqueeze(1)
            out_tmp = torch.cat((out_tmp, out_t ), 1)
        out = out_tmp
        if self.DEBUG:
            print(out.device)

        # input shape: [batch_size, seq_len, C, H, W]
        # out: list of tensors with shape [batch_size, seq_len, C, H, W], each tensor belongs to one time step
        out, _  = self.lstm(out)
        if self.DEBUG:
            print(out.device)

        # apply conv1x1 element-wise
        out_tmp = torch.tensor([]).to(self.device)
        for i in range(seq_len):
            out_t = self.conv1x1(out[:, i, :, :, :])  # extract base features of shape [seq_len*batch_size, 196, ceil(H/16), ceil(W/16)]
            out_t  = out_t.unsqueeze(1)
            out_tmp = torch.cat((out_tmp, out_t ), 1)
        out = out_tmp
        if self.DEBUG:
            print(out.device)

        # apply PS ROI pooling to crop features inside bounding boxes
        out_tmp = torch.tensor([]).to(self.device)
        for i in range(seq_len):
            out_t = self.ps_roi_pool(out[:, i, :, :, :], boxes_prev[:, i, :, :].contiguous().view(-1, 5))  # extract base features of shape [seq_len*batch_size, 196, ceil(H/16), ceil(W/16)]
            #print("out", out_t.shape)
            out_t  = out_t.unsqueeze(1)
            out_tmp = torch.cat((out_tmp, out_t ), 1)
        out = out_tmp
        if self.DEBUG:
            print(out.device)

        out_tmp = torch.tensor([]).to(self.device)
        for i in range(seq_len):
            out_t = self.pooling(out[:, i, :, :, :])  # extract base features of shape [seq_len*batch_size, 196, ceil(H/16), ceil(W/16)]
            out_t  = out_t.unsqueeze(1)
            out_tmp = torch.cat((out_tmp, out_t ), 1)
        out = out_tmp
        if self.DEBUG:
            print(out.device)

        out = out.squeeze()
        if self.DEBUG:
            print(out.device)

        out = out.view(batch_size, seq_len, num_boxes, 4)
        if self.DEBUG:
            print(out.device)

        # pick out the last velocity for current time step
        out = out[:, -1, :, :]
        if self.DEBUG:
            print(out.device)

        return out  # shape [batch_size, 1, num_boxes, 4] with row format [vxc, vyc, vw, vh]
