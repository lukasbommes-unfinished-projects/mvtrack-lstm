import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision

from lib.model.resnet_atrous import resnet18
from lib.utils import load_pretrained_weights_to_modified_resnet, \
    load_pretrained_weights, count_params, weight_checksum


class TrackNetOTCD(nn.Module):
    def __init__(self, pooling_size=7):
        super(TrackNetOTCD, self).__init__()

        self.FIXED_BLOCKS = 1
        self.pooling_size = pooling_size
        self.base_scale = 1/16

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
        self.pooling = nn.AvgPool2d(kernel_size=self.pooling_size, stride=self.pooling_size)
        self.ps_roi_pool = torchvision.ops.PSRoIPool(output_size=(self.pooling_size, self.pooling_size), spatial_scale=self.base_scale)

        assert (0 <= self.FIXED_BLOCKS <= 4) # set this value to 0, so we can train all blocks
        if self.FIXED_BLOCKS >= 4: # fix all blocks
            for p in self.RCNN_base[10].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 3: # fix first 3 blocks
            for p in self.RCNN_base[8].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 2: # fix first 2 blocks
            for p in self.RCNN_base[6].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 1: # fix first 1 block
            for p in self.RCNN_base[4].parameters(): p.requires_grad = False

    def forward(self, mvs_residuals, boxes_prev):
        # mvs_residuals is a tensor of shape [batch_size, 5, H, W] with C 0:2 MV XY and C 2:5 Residual BGR
        # boxes_prev is a tensor of shape [batch_size, num_boxes, 5] with row format [frame_idx, x1, x2, y1, y2]
        x = self.RCNN_base(mvs_residuals)
        x = self.RCNN_bbox_base(x)
        x = self.ps_roi_pool(x, boxes_prev.view(-1, 5))
        x = self.pooling(x)
        x = x.squeeze()
        return x
