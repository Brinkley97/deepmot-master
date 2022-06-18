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
from torch.utils import data
import os
import numpy as np


def prepare_Data(data_pth, train=True):
    """
    :param data_pth: string that gives the data path
    :return: data list
    """
    data = list()
    if train:
        d_pth = os.path.join(data_pth, 'train/')
    else:
        d_pth = os.path.join(data_pth, 'test/')
    dirs = os.listdir(d_pth)
    for dir in dirs:
        pth = os.path.join(d_pth, dir)
        files = os.listdir(pth)
        for file in files:
            if '_m.npy' in file:
                data.append(os.path.join(pth, file))
    return data


class RealData(data.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, data_path, train=True):
        'Initialization'
        self.data_pth = data_path
        self.data = prepare_Data(data_path, train)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        m_pth = self.data[index]  # input matrix name
        split = m_pth.split('_')
        t_pth = '_'.join(split[:-1])+'_t.npy'

        # Load data and get label
        matrix = np.load(m_pth)
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))

        # random permutations
        # vertical
        if torch.rand(1).item() < 0.5:
            idx = torch.randperm(matrix.shape[1])
            matrix = matrix[0:, idx, 0:]
            target = target[0:, idx, 0:]

        # horizontal
        if torch.rand(1).item() < 0.5:
            idx = torch.randperm(matrix.shape[2])
            matrix = matrix[0:, 0:, idx]
            target = target[0:, 0:, idx]

        return [matrix, target]