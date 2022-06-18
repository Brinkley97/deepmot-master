#==========================================================================
# This file is under License LGPL-3.0 (see details in the license file).
# This file is a part of implementation for paper:
# How To Train Your Deep Multi-Object Tracker.
# This contribution is headed by Perception research team, INRIA.
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
# created on 16th April 2020.
#==========================================================================
import torch
def weighted_binary_focal_entropy(output, target, weights=None, gamma=2):
    # output = torch.clamp(output, min=1e-8, max=1 - 1e-8)
    if weights is not None:
        assert weights.size(1) == 2

        # weight is of shape [batch,2, 1, 1]
        # weight[:,1] is for positive class, label = 1
        # weight[:,0] is for negative class, label = 0

        loss = (torch.pow(1.0-output, gamma)*torch.mul(target, weights[:, 1]) * torch.log(output+1e-8)) + \
               (torch.mul((1.0 - target), weights[:, 0]) * torch.log(1.0 - output+1e-8)*torch.pow(output, gamma))
    else:
        loss = target * torch.log(output+1e-8) + (1 - target) * torch.log(1 - output+1e-8)

    return torch.neg(torch.mean(loss))

