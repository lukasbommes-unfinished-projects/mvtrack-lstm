import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision

from lib.model.resnet_atrous import resnet18
from lib.model.convrnn import Conv2dGRU
from lib.utils import load_pretrained_weights_to_modified_resnet, \
    load_pretrained_weights, count_params, weight_checksum


class BaseNet(nn.Module):
    def __init__(self, pooling_size):
        super(BaseNet, self).__init__()

        self.FIXED_BLOCKS = 0

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

        self.RCNN_bbox_base = nn.Conv2d(512, 4*pooling_size*pooling_size, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)

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
        x = self.RCNN_bbox_base(x)
        return x


class TrackNet(nn.Module):
    def __init__(self, pooling_size=7):
        super(TrackNet, self).__init__()

        self.DEBUG = False
        self.device = None
        self.pooling_size = pooling_size  # the ROIs are split into m x m regions
        self.base_scale = 1/16  # ration of base features to input size

        self.base = BaseNet(pooling_size=pooling_size)
        # load OTCD weigths into base net
        weights_file = "models/OTCD_trackingnet.pth"
        state_dict = torch.load(weights_file)
        self.base.load_state_dict(state_dict["model"])
        print("Loaded weights from {} into BaseNet.".format(weights_file))
        # fix base net weights
        for p in self.base.parameters(): p.requires_grad = False

        num_in_channels = 4*pooling_size*pooling_size
        self.lstm = Conv2dGRU(in_channels=num_in_channels,
                               out_channels=num_in_channels,
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
        #self.conv1x1 = nn.Conv2d(196, 4*self.pooling_size*self.pooling_size, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.ps_roi_pool = torchvision.ops.PSRoIPool(output_size=(self.pooling_size, self.pooling_size), spatial_scale=self.base_scale)

        if self.DEBUG:
            print("base net param count: ", count_params(self.base))
            print("lstm param count: ", count_params(self.lstm))
            #print("conv1x1 param count: ", count_params(self.conv1x1))


    def forward(self, mvs_residuals, boxes_prev):
        # mvs_residuals is a tensor of shape [batch_size, seq_len, 5, H, W] with C 0:2 MV XY and C 2:5 Residual BGR
        # boxes_prev is a tensor of shape [batch_size, seq_len, num_boxes, 5] with row format [frame_idx, x1, x2, y1, y2]
        if self.device is None:
            raise RuntimeError("Please set the device attribute explicitely.")

        batch_size, seq_len, C, H, W = mvs_residuals.shape
        num_boxes = boxes_prev.shape[-2]

        mvs_residuals = mvs_residuals.view(batch_size * seq_len, C, H, W)
        out = self.base(mvs_residuals)

        out = out.view(batch_size, seq_len, *out.shape[1:])
        out, _ = self.lstm(out)
        out = out.contiguous().view(batch_size * seq_len, *out.shape[2:])

        boxes_prev = boxes_prev.view(batch_size * seq_len * num_boxes, 5)
        out = self.ps_roi_pool(out, boxes_prev)
        out = self.pooling(out)
        out = out.squeeze()
        out = out.view(batch_size, seq_len, num_boxes, 4)
        out = out.mean(1)

        return out  # shape [batch_size, num_boxes, 4] with row format [vxc, vyc, vw, vh]
