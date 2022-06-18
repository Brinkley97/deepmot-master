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
# Faster R-CNN Written by Ross Girshick and Xinlei Chen,
# MIT licence, Copyright (c) 2015 Microsoft
#==========================================================================
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from ..model.bbox_transform import bbox_transform_inv, clip_boxes
from ..model.config import cfg
from ..model.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))
  scores = scores.contiguous().view(-1, 1)
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  scores, order = scores.view(-1).sort(descending=True)
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
    scores = scores[:pre_nms_topN].view(-1, 1)
  proposals = proposals[order.data, :]

  # Non-maximal suppression
  keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep,]

  # Only support single image as input
  batch_inds = Variable(proposals.data.new(proposals.size(0), 1).zero_())
  blob = torch.cat((batch_inds, proposals), 1)

  return blob, scores
