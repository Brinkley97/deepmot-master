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
# Fast R-CNN Written by Ross Girshick, MIT Licence, Copyright (c) 2015 Microsoft
#==========================================================================
from torch.nn.modules.module import Module
from ..functions.roi_crop import RoICropFunction

class _RoICrop(Module):
    def __init__(self, layout = 'BHWD'):
        super(_RoICrop, self).__init__()
    def forward(self, input1, input2):
        return RoICropFunction()(input1, input2)
