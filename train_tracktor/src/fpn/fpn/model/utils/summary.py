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
# https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
#==========================================================================
import os

def write_scalars(writer, scalars, names, n_iter, tag=None):
    for i, scalar, in enumerate(scalars):
        if tag is not None:
            name = os.path.join(tag, names[i])
        else:
            name = names[i]
        writer.add_scalar(name, scalar, n_iter)

def write_hist_parameters(writer, net, n_iter):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

