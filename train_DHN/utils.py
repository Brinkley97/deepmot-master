#==========================================================================
# This file is under License LGPL-3.0 (see details in the license file).
# This file is a part of implementation for paper:
# How To Train Your Deep Multi-Object Tracker.
# This contribution is headed by Perception research team, INRIA.
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
# created on 16th April 2020.
#==========================================================================

import motmetrics as mm
import numpy as np
import csv
import copy
import torch
import shutil

persons_class = ["1"]


def threshold(T, matrix, value=np.NaN, mask=None):

    if value is not np.NaN:
        if mask is None:
            mask = np.zeros_like(matrix)
            mask[np.where(matrix > T)] = 1.0
        matrix = value*mask*matrix + (1.0-mask)*matrix

    else:
        if mask is None:
            matrix[np.where(matrix > T)] = value
        else:
            matrix[np.where(mask==1.0)] = value

    return matrix


def thresholdAdd(T, matrix, value=np.NaN, mask=None):

    if value is not np.NaN:
        if mask is None:
            mask = np.zeros_like(matrix)
            mask[np.where(matrix > T)] = 1.0
        mask_a = mask.copy()* value
        matrix += mask_a.copy()

    else:
        if mask is None:
            matrix[np.where(matrix > T)] = value
        else:
            matrix[np.where(mask==1.0)] = value

    return matrix


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


def getTarget(matrix, T, mask=None):
    """
    :param matrix: threshold distance matrix numpy array of shape [batch, H, W]
    :param T: threshold
    :return: target after munkres algorithm {0,1} of shape [batch, H, W]
    """

    matrix = threshold(T, matrix.copy(), mask=mask)
    targets = np.zeros_like(matrix)
    # for each batch
    for i in range(targets.shape[0]):
        acc = mm.MOTAccumulator(auto_id=True)
        acc.update(
            range(matrix.shape[2]),  # number of objects = matrix width
            range(matrix.shape[1]),  # number of hypothesis = matrix height
            np.transpose(matrix[i, :, :])
        )
        res = acc.mot_events.values
        for j in range(res.shape[0]):
            if res[j][0] == 'MATCH':
                targets[i, res[j][2], res[j][1]] = 1

    return targets


def read_txt_det(textpath):
    """
    :param textpath: string text path
    :return: a dict with key = frameid and value is a list of lists [object id, x, y, w, h] in the frame
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        # headers = next(f_csv)
        frames = {}
        for line in f_csv:
            if len(line) <=7:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            frames[line[0]].append(line[1:6])
    return frames


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


def read_txt_gtV2(textpath):
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
            if len(line) < 7 or (line[7] not in persons_class and "MOT2015" not in textpath) or int(float(line[6]))==0:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([line[1]]+bbox)
        f.close()
    ordered = reorder_frameID(frames)
    return ordered


def read_txt_detV4(textpath, th):
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
            if len(line) < 7:
                print(line)
                continue
            # score < 0.4, height < 0.0 frameid < 1
            if float(line[6]) < th or float(line[5]) < 0 or float(line[0]) < 1:
                continue

            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            # bbox += [float(line[6])]
            frames[line[0]].append(bbox)
            # f.close()
        ordered = reorder_frameID(frames)
        return ordered


def GIOU(boxes1, boxes2):
    boxes2 = torch.FloatTensor(boxes2).cuda() # gt box
    boxes1 = boxes1.unsqueeze(0)
    # print(boxes2.shape)
    # print(boxes1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=1)

    # print("boxes2", boxes2.shape)
    # print("boxes1", boxes1.shape)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # print("boxes1_area",boxes1_area.shape)
    # print("boxes2_area",boxes2_area.shape)

    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # print("intersection_left_up", intersection_left_up.shape)
    # print("intersection_right_down", intersection_right_down.shape)

    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros(1).cuda())
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    # print("IOU", IOU.shape)

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros(1).cuda())
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area

    return GIOU


def calculate_distV2_fast(bbox_det, bbox_gt, im_h, im_w):
    """
    :param bbox_det: one detection bbox [x, y, x, y]
    :param bbox_gt: one ground truth bbox [x, y, x, y]
    :param im_h: image height
    :param im_w: image width
    :return: normalized euclidean distance between detection and ground truth
    """
    x1_det, y1_det, x2_det, y2_det = bbox_det[0], bbox_det[1], bbox_det[2], bbox_det[3]
    x1_gt, y1_gt, x2_gt, y2_gt = bbox_gt[:, 0], bbox_gt[:, 1], bbox_gt[:, 2], bbox_gt[:, 3]

    D = (float(im_w)**2 + im_h**2)**0.5
    c_gt_x, c_gt_y = 0.5*(x1_gt + x2_gt), 0.5*(y1_gt + y2_gt)
    c_det_x, c_det_y = 0.5*(x1_det + x2_det), 0.5*(y1_det + y2_det)

    return 1.0 - np.exp(
        -5.0 * np.sqrt(1e-12 + np.power((c_gt_x - c_det_x) / D, 2.0) + np.power((c_gt_y - c_det_y) / D, 2.0)))


def calculate_distl2old_fast(bbox_det, bbox_gt, im_h, im_w):
    """
    :param bbox_det: one detection bbox [x, y, x, y]
    :param bbox_gt: ground truth bboxes [x, y, x, y] of shape [N,4]
    :param im_h: image height
    :param im_w: image width
    :return: normalized euclidean distance between detection and ground truth
    """

    x1_det, y1_det, x2_det, y2_det = bbox_det[0], bbox_det[1], bbox_det[2], bbox_det[3]
    x1_gt, y1_gt, x2_gt, y2_gt = bbox_gt[:, 0], bbox_gt[:, 1], bbox_gt[:, 2], bbox_gt[:, 3]

    c_gt_x, c_gt_y = 0.5*(x1_gt + y2_gt)/im_w, 0.5 * (y1_gt + y2_gt)/im_h
    c_det_x, c_det_y = 0.5 * (x1_det + y2_det)/im_w, 0.5 * (y1_det + y2_det)/im_h

    return np.sqrt(1e-12 + np.power((c_gt_x - c_det_x), 2.0) + np.power((c_gt_y - c_det_y), 2.0))


def make_matrix_l2iou(gt_bboxes, det_bboxes, img_h, img_w, threshold_flag=False, IOU_mask=None):
    """
    :param gt_bboxes: np array of ground truth bboxes [id, x, y, h, w] from img of frameID
    :param det_bboxes: np array of detection bboxes [id, x, y, h, w] from img of frameID
    :param frameID: ID of this frame
    :param img_h: height of the image
    :param img_w: width of the image
    :return: matrices and corresponding targets to be saved in a list
    """
    # number of detections = N = height of matrix
    N = det_bboxes.shape[0]
    # number of ground truth objects = M = width of matrix
    M = gt_bboxes.shape[0]

    assert gt_bboxes.shape[1] == 4
    assert det_bboxes.shape[1] == 4

    dist_mat = np.zeros((N, M), dtype=np.float32)
    for i in range(N):

        dist_mat[i, :] = 0.0 + 0.5*(calculate_distV2_fast(det_bboxes[i, :], gt_bboxes, img_h, img_w) +
                                        1.0-bb_fast_IOU_v1(det_bboxes[i, :], gt_bboxes))

    # threshold for data augmentation x 5
    matrices = []
    targets = []
    # without threshold
    T = 100.0
    matrices.append(dist_mat[np.newaxis].copy())
    target = getTarget(dist_mat[np.newaxis], T)
    targets.append(target.copy())

    # iou threshold, iou = 0.5, iou = 0.7
    mask = np.zeros_like(dist_mat.copy())
    mask[np.where(IOU_mask < 0.5)] = 1.0
    matrix = thresholdAdd(T, dist_mat.copy(), 1000, mask=mask)[np.newaxis]
    target = getTarget(dist_mat[np.newaxis], T, mask=mask[np.newaxis])
    matrices.append(matrix)
    targets.append(target)

    if threshold_flag:
        # with threshold
        T = 0.0
        old_T = [0.5, 0.0]
        candidates = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        for i in range(3):
            while T in old_T:
                # uniform distribution [0, 1]
                T = np.random.choice(candidates)
            old_T.append(T)
            mask = np.zeros_like(dist_mat.copy())
            mask[np.where(IOU_mask < T)] = 1.0
            target = getTarget(dist_mat[np.newaxis], T, mask=mask[np.newaxis])
            matrix = thresholdAdd(T, dist_mat.copy(), 1000, mask=mask)[np.newaxis]

            matrices.append(matrix)
            targets.append(target)
    return matrices, targets


def make_matrix_l2(gt_bboxes, det_bboxes, img_h, img_w, threshold_flag=False, new=False, IOU_mask=None):
    """
    :param gt_bboxes: list of ground truth bboxes [id, x, y, h, w] from img of frameID
    :param det_bboxes: list of detection bboxes [id, x, y, h, w] from img of frameID
    :param frameID: ID of this frame
    :param img_h: height of the image
    :param img_w: width of the image
    :return: matrices and corresponding targets to be saved in a list
    """
    # number of detections = N = height of matrix
    N = det_bboxes.shape[0]
    # number of ground truth objects = M = width of matrix
    M = gt_bboxes.shape[0]

    assert gt_bboxes.shape[1] == 4
    assert det_bboxes.shape[1] == 4

    dist_mat = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        if not new:
            dist_mat[i, :] = 0.0 + calculate_distl2old_fast(det_bboxes[i, :], gt_bboxes, img_h, img_w)
        else:
            dist_mat[i, :] = 0.0 + calculate_distV2_fast(det_bboxes[i, :], gt_bboxes, img_h, img_w)

    # threshold for data augmentation x 5
    matrices = []
    targets = []
    # without threshold
    T = 100.0
    matrices.append(dist_mat[np.newaxis].copy())
    target = getTarget(dist_mat[np.newaxis], T)
    targets.append(target.copy())

    # iou threshold, iou = 0.5, iou = 0.7
    mask = np.zeros_like(dist_mat.copy())
    mask[np.where(IOU_mask < 0.5)] = 1.0
    matrix = thresholdAdd(T, dist_mat.copy(), 1000, mask=mask)[np.newaxis]
    target = getTarget(dist_mat[np.newaxis], T, mask=mask[np.newaxis])
    matrices.append(matrix)
    targets.append(target)

    if threshold_flag:
        # with threshold
        T = 0.0
        old_T = [0.5, 0.0]
        candidates = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        for i in range(3):
            while T in old_T:
                # uniform distribution [0, 1]
                T = np.random.choice(candidates)
            old_T.append(T)
            mask = np.zeros_like(dist_mat.copy())
            mask[np.where(IOU_mask < T)] = 1.0
            target = getTarget(dist_mat[np.newaxis], T, mask=mask[np.newaxis])
            matrix = thresholdAdd(T, dist_mat.copy(), 1000, mask=mask)[np.newaxis]

            matrices.append(matrix)
            targets.append(target)
    return matrices, targets


def make_matrix_giou(gt_bboxes, det_bboxes, threshold_flag=False, IOU_mask=None):
    """
    :param gt_bboxes: list of ground truth bboxes [x, y, h, w] from img of frameID
    :param det_bboxes: list of detection bboxes [id, x, y, h, w] from img of frameID
    :param frameID: ID of this frame
    :param img_h: height of the image
    :param img_w: width of the image
    :return: matrices and corresponding targets to be saved in a list
    """
    # number of detections = N = height of matrix
    N = det_bboxes.shape[0]
    # number of ground truth objects = M = width of matrix
    M = gt_bboxes.shape[0]

    assert gt_bboxes.shape[1] == 4
    assert det_bboxes.shape[1] == 4

    dist_mat = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        det = torch.FloatTensor(det_bboxes[i, :]).cuda()
        dist_mat[i, :] = 0.5*(1.0-GIOU(det, gt_bboxes).cpu().numpy())

    # threshold for data augmentation x 5
    matrices = []
    targets = []
    # without threshold
    T = 100.0
    matrices.append(dist_mat[np.newaxis].copy())
    target = getTarget(dist_mat[np.newaxis], T)
    targets.append(target.copy())

    # iou threshold, iou = 0.5, iou = 0.7
    mask = np.zeros_like(dist_mat.copy())
    mask[np.where(IOU_mask < 0.5)] = 1.0
    matrix = thresholdAdd(T, dist_mat.copy(), 1000, mask=mask)[np.newaxis]
    target = getTarget(dist_mat[np.newaxis], T, mask=mask[np.newaxis])
    matrices.append(matrix)
    targets.append(target)

    if threshold_flag:
        # with threshold
        T = 0.0
        old_T = [0.5, 0.0]
        candidates = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        for i in range(3):
            while T in old_T:
                # uniform distribution [0, 1]
                T = np.random.choice(candidates)
            old_T.append(T)
            mask = np.zeros_like(dist_mat.copy())
            mask[np.where(IOU_mask < T)] = 1.0
            target = getTarget(dist_mat[np.newaxis], T, mask=mask[np.newaxis])
            matrix = thresholdAdd(T, dist_mat.copy(), 1000, mask=mask)[np.newaxis]

            matrices.append(matrix)
            targets.append(target)
    return matrices, targets


# deep munkres utils #
def adjust_learning_rate(optimizer, iteration, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.95 ** (iteration // 20000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_model_name= ""):
    torch.save(state, filename)
    torch.save(state['state_dict'], filename[:-4])
    if is_best:
        shutil.copyfile(filename, best_model_name)
        shutil.copyfile(filename[:-4], best_model_name[:-4])


def eval_acc(score, target, weight, th=0.5):
    """
    :param score: torch tensor, predicted score of shape [batch, H, W]
    :param target: torch tensor, ground truth value {0,1} of shape [batch, H, W]
    :param weight: torch tensor, weight for each batch for negative and positive examples of shape [batch, 2, 1, 1]
    :return: accuracy
    """
    acc = []
    predicted = torch.zeros_like(score).cuda().float()
    for b in range(score.size(0)):
        for h in range(score.size(1)):
            value, indice = score[b, h].max(0)
            if float(value) > th:
                predicted[b, h, int(indice)] = 1.0
        num_positive = float(target[b, :, :].sum())
        num_negative = float(target.size(1)*target.size(2) - num_positive)
        num_tp = float(((predicted[b, :, :] == target[b, :, :]).float() + (target[b, :, :] == 1.0).float()).eq(2).sum())
        num_tn = float(((predicted[b, :, :] == target[b, :, :]).float() + (target[b, :, :] == 0.0).float()).eq(2).sum())
        acc.append(1.0*(num_tp * float(weight[b, 1, 0, 0]) + num_tn * float(weight[b, 0, 0, 0]))/
                   (num_positive * float(weight[b, 1, 0, 0]) + num_negative * float(weight[b, 0, 0, 0])))

    return predicted, np.mean(np.array(acc))



