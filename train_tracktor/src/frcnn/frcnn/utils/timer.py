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
# Faster R-CNN Written by Ross Girshick,
# MIT licence, Copyright (c) 2015 Microsoft
#==========================================================================


import time
import torch

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self._total_time = {}
        self._calls = {}
        self._start_time = {}
        self._diff = {}
        self._average_time = {}

    def tic(self, name='default'):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        torch.cuda.synchronize()
        self._start_time[name] = time.time()

    def toc(self, name='default', average=True):
        torch.cuda.synchronize()
        self._diff[name] = time.time() - self._start_time[name]
        self._total_time[name] = self._total_time.get(name, 0.) + self._diff[name]
        self._calls[name] = self._calls.get(name, 0 ) + 1
        self._average_time[name] = self._total_time[name] / self._calls[name]
        if average:
            return self._average_time[name]
        else:
            return self._diff[name]

    def average_time(self, name='default'):
        return self._average_time[name]

    def total_time(self, name='default'):
        return self._total_time[name]

timer = Timer()
