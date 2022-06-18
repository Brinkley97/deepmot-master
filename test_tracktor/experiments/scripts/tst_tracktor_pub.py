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
import sys
root_pth = '/'.join(osp.dirname(__file__).split('/')[:-2])
sys.path.insert(1, root_pth)
from sacred import Experiment
from src.tracktor.datasets.factory import Datasets
from src.tracktor.resnet import resnet50
from src.tracktor.tracker import Tracker
from src.tracktor.utils import interpolate, plot_img

ex = Experiment()

ex.add_config(root_pth+'/experiments/cfgs/tracktor_pub.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config('/'.join(osp.dirname(__file__).split('/')[:-3]) + ex.configurations[0]._conf['tracktor']['reid_network_config'])
ex.add_config(root_pth + ex.configurations[0]._conf['tracktor']['obj_detect_config'])

# Tracker = ex.capture(Tracker, prefix='tracker.tracker')

@ex.automain
def my_main(tracktor, siamese, _config):
    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    # output dir
    output_dir = root_pth + osp.join(tracktor['output_dir'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    # color list for plotting
    def get_spaced_colors(n):
        max_value = 16581375  # 255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    colorList = get_spaced_colors(100)
    np.random.shuffle(colorList)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("[*] Building object detector")
    if tracktor['network'].startswith('fpn'):
        # FPN
        from src.tracktor.fpn import FPN
        from src.fpn.fpn.model.utils import config
        config.cfg.TRAIN.USE_FLIPPED = False
        config.cfg.CUDA = True
        config.cfg.TRAIN.USE_FLIPPED = False
        print("load weights from: ", '/'.join(osp.dirname(__file__).split('/')[:-3]) + tracktor['obj_detect_weights'])
        checkpoint = torch.load('/'.join(osp.dirname(__file__).split('/')[:-3]) + tracktor['obj_detect_weights'])

        if 'pooling_mode' in checkpoint.keys():
            config.cfg.POOLING_MODE = checkpoint['pooling_mode']
        else:
            config.cfg.POOLING_MODE = 'align'

        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                    'ANCHOR_RATIOS', '[0.5,1,2]']
        config.cfg_from_file(root_pth + _config['tracktor']['obj_detect_config'])
        config.cfg_from_list(set_cfgs)

        if 'fpn_1_12.pth' in tracktor['obj_detect_weights']:
            classes = ('__background__',
             'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor')
        else:
            classes = ('__background__', 'pedestrian')

        obj_detect = FPN(classes, 101, pretrained=False)
        obj_detect.create_architecture()
        if 'model' in checkpoint.keys():
            obj_detect.load_state_dict(checkpoint['model'], strict=False)
        else:
            obj_detect.load_state_dict(checkpoint, strict=False)


    else:
        raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")

    pprint.pprint(tracktor)
    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **siamese['cnn'])
    reid_network.load_state_dict(torch.load('/'.join(osp.dirname(__file__).split('/')[:-3]) + tracktor['reid_network_weights']))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    print("[*] Beginning evaluation...")

    time_total = 0
    for sequence in Datasets(tracktor['dataset']):
        tracker.reset()

        if "15" in tracktor['dataset']:
            if os.path.exists(os.path.join(output_dir, sequence._seq_name + '.txt')):
                continue
        elif '16' in tracktor['dataset']:
            print("[*] Evaluating: {}".format('MOT16-' + sequence._seq_name[6:8]))
            if os.path.exists(osp.join(output_dir, 'MOT16-' + sequence._seq_name[6:8] + '.txt')):
                print('MOT16-' + sequence._seq_name[6:8] + ' exists, skip.')
                continue
        elif '17' in tracktor['dataset']:
            print("[*] Evaluating: {}".format('MOT17-' + sequence._seq_name[6:8] + "-" + sequence._dets[:-2]))
            if os.path.exists(
                    osp.join(output_dir, 'MOT17-' + sequence._seq_name[6:8] + "-" + sequence._dets[:-2] + '.txt')):
                print('MOT17-' + sequence._seq_name[6:8] + "-" + sequence._dets[:-2] + ' exists, skip.')



        now = time.time()

        print("[*] Evaluating: {}".format(sequence))

        data_loader = DataLoader(sequence, batch_size=1, shuffle=False)
        for i, frame in enumerate(data_loader):
            frame["im_info"] = frame["im_info"].cuda()
            frame["app_data"] = frame["app_data"].cuda()
            print("frame: ", frame['im_path'][0].split("/")[-1])
            if i >= len(sequence) * tracktor['frame_split'][0] and i <= len(sequence) * tracktor['frame_split'][1]:
                tracker.step_pub(frame)
        results = tracker.get_results()

        time_total += time.time() - now

        print("[*] Tracks found: {}".format(len(results)))
        print("[*] Time needed for {} evaluation: {:.3f} s".format(sequence, time.time() - now))

        if tracktor['interpolate']:
            results = interpolate(results)

        sequence.write_results(results, osp.join(output_dir))

        if tracktor['write_images']:
            if '16' in tracktor['dataset']:
                plot_img(tracktor['data_pth'], "MOT16", osp.join(output_dir, 'MOT16-'+sequence._seq_name[6:8]+'.txt'),
                         output_dir+'/plot/', 'MOT16-'+sequence._seq_name[6:8], 'MOT16-'+sequence._seq_name[6:8], colorList)
            elif '17' in tracktor['dataset']:
                plot_img(tracktor['data_pth'], "MOT17", osp.join(output_dir, sequence._seq_name + "-" + sequence._dets[:-2] + '.txt'),
                        output_dir+'/plot/', sequence._seq_name, sequence._seq_name + "-" + sequence._dets[:-2], colorList)


    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_total))
