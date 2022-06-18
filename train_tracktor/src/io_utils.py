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

import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import csv
import cv2
import sys
import copy
import motmetrics
import os

persons_class = ["1"]

def xywh2xyxy(bbox):
    """
    :param bbox: bbox in string [x, y, w, h]
    :return: bbox in float [x1, y1, x2, y2]
    """
    copy.deepcopy(bbox)
    bbox[0] = float(bbox[0])
    bbox[1] = float(bbox[1])
    bbox[2] = float(bbox[2]) + bbox[0]
    bbox[3] = float(bbox[3]) + bbox[1]

    return bbox


def reorder_frameID(frame_dict):
    """
    reorder the frames dictionary in a ascending manner

    :param frame_dict: a dict with key = frameid and value is a list of lists [object id, x, y, w, h] in the frame
    :return: ordered frame dict
    """
    keys_int = sorted([int(i) for i in frame_dict.keys()])

    new_dict = {}
    for key in keys_int:
        new_dict[str(key)] = frame_dict[str(key)]
    return new_dict


def read_txt_gtV2(textpath, Tofilter=False):
    """
    :param textpath: string text path
    :return: a dict with key = frameid and value is a list of lists [object id, x1, y1, x2, y2] in the frame
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        # headers = next(f_csv)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            # we only consider "pedestrian" class #
            if len(line) < 7 or (line[7] not in persons_class and "MOT15" not in textpath) or \
                    int(float(line[6])) == 0 or (float(line[8]) < 0.5 and Tofilter):
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([line[1]]+bbox)
        f.close()
    ordered = reorder_frameID(frames)
    return ordered

def read_txt_predictionV2(textpath):
    """
    :param textpath: string text path
    :return: a dict with key = frameid and value is a list of lists [x1, y1, x2, y2] in the frame
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        # headers = next(f_csv)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            if len(line) <= 5:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([int(line[1])] + bbox)
        # f.close()
    ordered = reorder_frameID(frames)
    return ordered

def bb_fast_IOU_v1(boxA, boxB):
    """
    Version Non differentiable
    :param boxA: numpy array [top left x, top left y, x2, y2]
    :param boxB: numpy array of [top left x, top left y, x2, y2], shape = [num_bboxes, 4]
    :return: IOU of two bounding boxes of shape [num_bboxes]
    """
    if type(boxA) is type([]):
        boxA = np.array(copy.deepcopy(boxA), dtype=np.float32)[-4:]
        boxB = np.array(copy.deepcopy(boxB), dtype=np.float32)[:, -4:]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[:, 0])
    yA = np.maximum(boxA[1], boxB[:, 1])
    xB = np.minimum(boxA[2], boxB[:, 2])
    yB = np.minimum(boxA[3], boxB[:, 3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0.0, xB - xA + 1) * np.maximum(0.0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou