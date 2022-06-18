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
from torch.utils.data import Dataset
import torch

from .mot_siamese import MOT_Siamese


class MOT_Siamese_Wrapper(Dataset):
	"""A Wrapper class for MOT_Siamese.

	Wrapper class for combining different sequences into one dataset for the MOT_Siamese
	Dataset.
	"""

	def __init__(self, split, dataloader):

		train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
				'MOT17-11', 'MOT17-13']

		self._dataloader = MOT_Siamese(None, split=split, **dataloader)

		for seq in train_folders:
			d = MOT_Siamese(seq, split=split, **dataloader)
			for sample in d.data:
				self._dataloader.data.append(sample)

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		return self._dataloader[idx]
