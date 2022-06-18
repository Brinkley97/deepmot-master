#==========================================================================
# This file is under License LGPL-3.0 (see details in the license file).
# This file is a part of implementation for paper:
# How To Train Your Deep Multi-Object Tracker.
# This contribution is headed by Perception research team, INRIA.
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
# created on 16th April 2020.
# the code is modified based on:
# https://github.com/phil-bergmann/tracking_wo_bnw/tree/iccv_19
# https://github.com/jwyang/faster-rcnn.pytorch/
#==========================================================================
from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ..functions.roi_align import RoIAlignFunction


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois, scale):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
                                scale)(features, rois)

class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois, scale):
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                                scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)

class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois, scale):
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                                scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)
