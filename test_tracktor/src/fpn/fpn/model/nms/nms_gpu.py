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
import torch
import numpy as np
from ._ext import nms
import pdb

def nms_gpu(dets, thresh):
	keep = dets.new(dets.size(0), 1).zero_().int()
	num_out = dets.new(1).zero_().int()
	nms.nms_cuda(keep, dets, num_out, thresh)
	keep = keep[:num_out[0]]
	return keep
