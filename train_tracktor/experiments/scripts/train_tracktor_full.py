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
#==========================================================================

from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pprint
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import copy
import shutil
from tensorboardX import SummaryWriter
import sys

root_pth = '/'.join(osp.dirname(__file__).split('/')[:-2])
sys.path.insert(1, root_pth)
from src.tracktor.datasets.factory import Datasets
from src.tracktor.tracker import Tracker




def my_main(tracktor, _config):
    ##########################
    #      set all seeds     #
    ##########################
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    ##########################
    #      set output dir    #
    ##########################
    output_dir = root_pth + osp.join(tracktor['output_dir'])
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    print("output dir: ", output_dir)

    ##########################
    # Initialize the modules #
    ##########################
    # object detector: FPN
    if tracktor['network'].startswith('fpn'):
        from src.tracktor.fpn import FPN
        from src.fpn.fpn.model.utils import config
        config.cfg.TRAIN.USE_FLIPPED = False
        config.cfg.CUDA = True
        config.cfg.TRAIN.USE_FLIPPED = False
        checkpoint = torch.load(root_pth + tracktor['obj_detect_weights'])

        if 'pooling_mode' in checkpoint.keys():
            config.cfg.POOLING_MODE = checkpoint['pooling_mode']

        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                    'ANCHOR_RATIOS', '[0.5,1,2]']
        config.cfg_from_file(root_pth + _config['tracktor']['obj_detect_config'])
        config.cfg_from_list(set_cfgs)

        obj_detect = FPN(('__background__', 'pedestrian'), 101, pretrained=False)
        obj_detect.create_architecture()
        print("loading model: ", root_pth + tracktor['obj_detect_weights'])

        model_state_dict = obj_detect.state_dict()

        model_state_dict.update(checkpoint['model'])

        obj_detect.load_state_dict(model_state_dict)

        # optimizer to train FPN
        lr = 0.0001
        for key, value in dict(obj_detect.named_parameters()).items():

            if 'RCNN_bbox_pred' not in key and "reid_branch" not in key:
                value.requires_grad = False
            else:
                value.requires_grad = True

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            obj_detect.apply(set_bn_eval)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, obj_detect.parameters()), lr=lr)

    else:
        raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")

    obj_detect.cuda()

    # dhn
    from src.DHN import Munkrs
    DHN = Munkrs(element_dim=1, hidden_dim=256, target_size=1,
                 bidirectional=True, minibatch=1, is_cuda=True,
                 is_train=False)

    DHN.load_state_dict(torch.load(root_pth+tracktor['dhn']))

    DHN.cuda()

    # parameters
    split_sequence = 2
    min_seq_length = 1

    # tracktor #

    tracker = Tracker(obj_detect, None, DHN, optimizer, tracktor['tracker'])

    starting_epoch = 1
    while os.path.exists(output_dir + "/" + "best_model_" + str(starting_epoch) + ".pth.tar"):
        starting_epoch += 1

    if os.path.exists(output_dir + "/" + "best_model_" + str(starting_epoch - 1) + ".pth.tar"):

        saved_model = torch.load(output_dir + "/" + "best_model_" + str(starting_epoch - 1) + ".pth.tar")
        tracker.obj_detect.load_state_dict(saved_model['model_weight'])
        tracker.iterations = saved_model['iterations']
        tracker.optimizer.load_state_dict(saved_model['optimizer'])
        tracker.old_loss = saved_model['loss']
        # create logs #
        print("log path: ", output_dir + '/logs/' + 'train_log')
    else:
        if os.path.exists(output_dir + '/logs/' + 'train_log'):
            shutil.rmtree(output_dir + '/logs/' + 'train_log')

    # create logs #
    tracker.mota_writer = SummaryWriter(output_dir + '/logs/train_log/mota')
    tracker.motp_writer = SummaryWriter(output_dir + '/logs/train_log/motp')
    tracker.clasf_writer = SummaryWriter(output_dir + '/logs/train_log/clasf')

    print("[*] Beginning evaluation...")

    collect_sequences = dict()
    collect_start_points = list()

    for sequence in Datasets(tracktor['dataset']):
        print("[*] Evaluating: {}".format(sequence))

        # random start points #
        start_points = list()  # index i to start during training
        seq_len = len(sequence)
        if seq_len < split_sequence*5:  # small sequences, no need to be split
            start_points.append(0)
        else:
            num_small_seqs = int((seq_len - 1) // split_sequence)  # start with zero
            for j in range(num_small_seqs + 1):
                start_points.append(sequence._seq_name + '_' + str(j * split_sequence))
            if seq_len - j * split_sequence <= min_seq_length:  # last sequence smaller than 50 frames
                start_points.pop()

        start_points = start_points[0::5]  # every 10 frames
        np.random.shuffle(start_points)
        collect_sequences[sequence._seq_name] = copy.deepcopy(sequence)
        collect_start_points += start_points


    for epoch in range(starting_epoch, 30):

        np.random.shuffle(collect_start_points)
        for start_pt in collect_start_points:
            tracker.reset()
            seq_name = '_'.join(start_pt.split('_')[:-1])
            start_point = int(start_pt.split('_')[-1])

            cropped_sequence = copy.deepcopy(collect_sequences[seq_name])
            cropped_sequence.data = cropped_sequence.data[start_point:]
            data_loader = DataLoader(cropped_sequence, batch_size=1, shuffle=False)

            for i, frame in enumerate(data_loader):

                frame["im_info"] = frame["im_info"].cuda()
                frame["app_data"] = frame["app_data"].cuda()
                if i < (5*split_sequence):
                    tracker.step_full_reid(frame, epoch, output_dir, is_start=(i == 0))

                else:
                    break

            del data_loader
            del cropped_sequence
            torch.cuda.empty_cache()


with open(root_pth + '/experiments/cfgs/tracktor_full.yaml', 'r') as f:
    tracktor = yaml.load(f)['tracktor']

with open(root_pth + '/output/fpn/res101/mot_2017_train/voc_init_iccv19/config.yaml', 'r') as f:
    obj_detector = yaml.load(f)

_config = dict()

_config['tracktor'] = tracktor
_config.update(obj_detector)
my_main(tracktor, _config)

