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

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function
from .mot import MOT17, MOT19CVPR

__sets = {}

# MOT17 dataset
mot17_splits = ['train', 'frame_val', 'frame_train', 'test', 'all']
# we generate 7 train/val splits for cross validation with single sequence val sets
# ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
for i in range(1, 8):
  mot17_splits += [f'seq_train_{i}']
  mot17_splits += [f'seq_val_{i}']
for year in ['2017']:
  for split in mot17_splits:
    name = f'mot_{year}_{split}'
    __sets[name] = (lambda split=split, year=year: MOT17(split, year))

# MOT19_CVPR dataset
mot19_cvpr_splits = ['train', 'frame_val', 'frame_train', 'test', 'all']
# we generate 4 train/val splits for cross validation with single sequence val sets
# ['CVPR-01', 'CVPR-02', 'CVPR-03', 'CVPR-05']
for i in range(1, 5):
  mot19_cvpr_splits += [f'seq_train_{i}']
  mot19_cvpr_splits += [f'seq_val_{i}']
for split in mot19_cvpr_splits:
  name = f'mot19_cvpr_{split}'
  __sets[name] = (lambda split=split: MOT19CVPR(split))