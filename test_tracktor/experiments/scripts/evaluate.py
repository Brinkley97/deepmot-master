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

import motmetrics
import numpy as np
import os
import os.path as osp
import sys
root_pth = '/'.join(osp.dirname(__file__).split('/')[:-2])
sys.path.insert(1, root_pth)
import src.io_utils as utils
mh = motmetrics.metrics.create()
import yaml

with open(root_pth + '/experiments/cfgs/tracktor_pub_reid.yaml', 'r') as f:
  tracktor = yaml.load(f)['tracktor']

threshold = 0.5
predt_pth = []

for predtpth in predt_pth:

    if '16' in predtpth:
        pth = tracktor['data_pth'] + "/MOT16Labels/train/"
    else:
        pth = tracktor['data_pth'] + "/MOT17Labels/train/"

    txtes = os.listdir(predtpth)

    print("##################")
    print(predtpth)
    print("##################")

    total_fn = 0
    total_fp = 0
    total_idsw = 0
    total_num_objects = 0
    total_matched = 0
    sum_distance = 0

    for txt in txtes:
        if "txt" not in txt:
            continue
        vname = txt[:-4]
        print(pth + vname + "/gt/gt.txt")

        if not os.path.exists(pth + vname + "/gt/gt.txt"):
            continue

        print(vname)

        acc = motmetrics.MOTAccumulator(auto_id=True)
        # load detections and gt bbox of this sequence

        frames_gt = utils.read_txt_gtV2(pth + vname + "/gt/gt.txt")
        if len(frames_gt.keys()) == 0:
            print("cannot load gts")

        frames_prdt = utils.read_txt_predictionV2(os.path.join(predtpth,txt))
        if len(frames_prdt.keys()) == 0:
            print("cannot load detections")

        # evaluations

        for frameid in frames_gt.keys():
            # print("frameid: ", int(frameid)+1)
            # get gt ids
            gt_bboxes = np.array(frames_gt[frameid], dtype=np.float32)
            gt_ids = gt_bboxes[:, 0].astype(np.int32).tolist()

            if frameid in frames_prdt.keys():
                #print("frameid", frameid)
                # get id track
                id_track = np.array(frames_prdt[frameid])[:, 0].astype(np.int32).tolist()
                # get a binary mask from IOU, 1.0 if iou < 0.5, else 0.0
                mask_IOU = np.zeros((len(frames_prdt[frameid]), len(frames_gt[frameid])))
                # distance matrix
                distance_matrix = []
                for i, bbox in enumerate(frames_prdt[frameid]):
                    iou = utils.bb_fast_IOU_v1(bbox, frames_gt[frameid])
                    # threshold
                    th = np.zeros_like(iou)
                    th[np.where(iou <= threshold)] = 1.0
                    mask_IOU[i, :] = th

                    # distance
                    distance_matrix.append(1.0-iou)

                distance_matrix = np.vstack(distance_matrix)

                distance_matrix[np.where(mask_IOU == 1.0)] = np.nan
                #print(distance_matrix)

                acc.update(
                    gt_ids,  # number of objects = matrix width
                    id_track,  # number of hypothesis = matrix height
                    np.transpose(distance_matrix)
                )
            # empty tracks
            else:
                acc.update(
                    gt_ids,  # number of objects = matrix width
                    [],      # number of hypothesis = matrix height
                    [[], []]
                )

        summary = mh.compute(acc, metrics=['motp', 'mota', 'num_false_positives', 'num_misses', 'num_switches',
                                           'num_objects', 'num_matches'], name='final')
        total_fp += float(summary['num_false_positives'].iloc[0])
        total_fn += float(summary['num_misses'].iloc[0])
        total_idsw += float(summary['num_switches'].iloc[0])
        total_num_objects += float(summary['num_objects'].iloc[0])
        total_matched += float(summary['num_matches'].iloc[0])
        sum_distance += float(summary['motp'].iloc[0])* float(summary['num_matches'].iloc[0])
        strsummary = motmetrics.io.render_summary(
            summary,
            formatters={'mota': '{:.2%}'.format},
            namemap={'motp': 'MOTP', 'mota': 'MOTA', 'num_false_positives': 'FP', 'num_misses': 'FN',
                     'num_switches': "ID_SW", 'num_objects': 'num_objects'}
        )
        print(strsummary)

    print("avg mota: {:.3f} %".format(100.0*(1.0-(total_idsw+total_fn+total_fp)/total_num_objects)))
    print("avg motp: {:.3f} %".format(100.0 * (1.0 - sum_distance / total_matched)))
    print("total fn: ", total_fn)
    print("total fp: ", total_fp)
    print("total idsw: ", total_idsw)
    print("total_num_objects: ", total_num_objects)

