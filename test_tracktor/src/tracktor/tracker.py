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
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable
import cv2
from src.frcnn.frcnn.model.nms_wrapper import nms

from .utils import bbox_overlaps, bbox_transform_inv, clip_boxes

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= np.asarray([[[102.9801, 115.9465, 122.7717]]])

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in [600]:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > 1000:
      im_scale = float(1000) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  num_images = len(ims)
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob

class Tracker():
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg):
        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.detection_person_thresh = tracker_cfg['detection_person_thresh']
        self.regression_person_thresh = tracker_cfg['regression_person_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model = tracker_cfg['motion_model']

        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']

        self.reset()

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
                                     new_det_features[i].view(1, -1),
                                     self.inactive_patience, self.max_features_num))
        self.track_num += num_new

    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores."""
        pos = self.get_pos()

        # regress
        _, scores, bbox_pred, rois = self.obj_detect.test_rois(pos)
        boxes = bbox_transform_inv(rois, bbox_pred)
        boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

        if scores.shape[1] > 2:
            # print("yoyoyo")
            pos = boxes[:, 15 * 4:(15 + 1) * 4]
            scores = scores[:, 15]
        else:
            pos = boxes[:, self.cl * 4:(self.cl + 1) * 4]
            scores = scores[:, self.cl]

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # t.prev_pos = t.pos
                t.pos = pos[i].view(1, -1)
        return torch.Tensor(s[::-1]).cuda()

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with provided detections."""
        new_det_features = self.reid_network.test_rois(blob['app_data'][0], new_det_pos / blob['im_info'][0][2]).data
        if len(self.inactive_tracks) >= 1 and self.do_reid:
            # print("doing reid")
            # calculate appearance distances
            dist_mat = []
            pos = []
            for t in self.inactive_tracks:  # for each inactive track, calculate its appearance distance with all new dets
                # print('i am here')
                # for feat in new_det_features:
                # 	print(t.test_features(feat.view(1,-1)).shape)
                dist_mat.append(torch.cat([t.test_features(feat.view(1, -1)) for feat in new_det_features], 1))
                pos.append(t.pos)
            if len(dist_mat) > 1:  # for all inactive tracks
                dist_mat = torch.cat(dist_mat, 0)
                pos = torch.cat(pos, 0)
            else:
                dist_mat = dist_mat[0]
                pos = pos[0]

            # calculate IoU distances between inactive track and new dets
            iou = bbox_overlaps(pos, new_det_pos)
            iou_mask = torch.ge(iou, self.reid_iou_threshold)
            iou_neg_mask = ~iou_mask
            # make all impossible assignements to the same add big value
            dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
            dist_mat = dist_mat.cpu().numpy()
            # print(new_det_features.shape)
            # print(dist_mat.shape)

            row_ind, col_ind = linear_sum_assignment(dist_mat)

            assigned = []
            remove_inactive = []
            for r, c in zip(row_ind, col_ind):
                if dist_mat[r, c] <= self.reid_sim_threshold:
                    t = self.inactive_tracks[r]
                    self.tracks.append(t)
                    t.count_inactive = 0
                    t.last_v = torch.Tensor([])
                    t.pos = new_det_pos[c].view(1, -1)
                    t.add_features(new_det_features[c].view(1, -1))
                    assigned.append(c)
                    remove_inactive.append(t)

            for t in remove_inactive:
                self.inactive_tracks.remove(t)

            keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
            if keep.nelement() > 0:
                new_det_pos = new_det_pos[keep]
                new_det_scores = new_det_scores[keep]
                new_det_features = new_det_features[keep]
            else:
                new_det_pos = torch.zeros(0).cuda()
                new_det_scores = torch.zeros(0).cuda()
                new_det_features = torch.zeros(0).cuda()
        return new_det_pos, new_det_scores, new_det_features

    def reid_new(self, blob, new_det_pos, new_det_scores):
        with torch.no_grad():
            """Tries to ReID inactive tracks with provided detections."""
            # self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])

            new_det_features = self.get_appearances_new(new_det_pos).detach()
            real_new_det_features = self.obj_detect.reid_branch(new_det_features).detach()
            if len(self.inactive_tracks) >= 1 and self.do_reid:
                # print("doing reid")
                # calculate appearance distances
                dist_mat = []
                pos = []
                for t in self.inactive_tracks:  # for each inactive track, calculate its appearance distance with all new dets
                    # print('i am here')
                    # for feat in new_det_features:
                    # 	print(t.test_features(feat.view(1,-1)).shape)
                    dist_mat.append(t.test_features_new(real_new_det_features, self.obj_detect).detach())
                    pos.append(t.pos)
                if len(dist_mat) > 1:  # for all inactive tracks
                    dist_mat = torch.cat(dist_mat, 0)

                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]
                # print(self.reid_sim_threshold)
                # print(dist_mat.shape)

                # calculate IoU distances between inactive track and new dets
                iou = bbox_overlaps(pos, new_det_pos)
                iou_mask = torch.ge(iou, self.reid_iou_threshold)
                iou_neg_mask = ~iou_mask
                # make all impossible assignements to the same add big value
                dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    # todo ablation threshold
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.last_v = torch.Tensor([])
                        t.pos = new_det_pos[c].view(1, -1)
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()
        return new_det_pos, new_det_scores, new_det_features

    def clear_inactive(self):
        """Checks if inactive tracks should be removed."""
        to_remove = []
        for t in self.inactive_tracks:
            if t.is_to_purge():
                to_remove.append(t)
        for t in to_remove:
            self.inactive_tracks.remove(t)

    def get_appearances(self, blob):
        """Uses the siamese CNN to get the features for all active tracks."""
        with torch.no_grad():
            new_features = self.reid_network.test_rois(blob['app_data'][0], self.get_pos() / blob['im_info'][0][2]).data
        return new_features

    def get_appearances_new(self, pos):
        """Uses the siamese CNN to get the features for all active tracks."""
        with torch.no_grad():
            new_features = self.obj_detect.extract_features(pos).detach()
        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = self.last_image.cpu().numpy()
            im2 = blob['data'][0][0].cpu().numpy()
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            sz = im1.shape
            warp_mode = self.warp_mode
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            # number_of_iterations = 5000
            number_of_iterations = self.number_of_iterations
            termination_eps = self.termination_eps
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
            (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
            warp_matrix = torch.from_numpy(warp_matrix)
            pos = []
            for t in self.tracks:
                p = t.pos[0]
                p1 = torch.Tensor([p[0], p[1], 1]).view(3, 1)
                p2 = torch.Tensor([p[2], p[3], 1]).view(3, 1)

                p1_n = torch.mm(warp_matrix, p1).view(1, 2)
                p2_n = torch.mm(warp_matrix, p2).view(1, 2)
                pos = torch.cat((p1_n, p2_n), 1).cuda()

                t.pos = pos.view(1, -1)
            # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            if self.do_reid:
                for t in self.inactive_tracks:
                    p = t.pos[0]
                    p1 = torch.Tensor([p[0], p[1], 1]).view(3, 1)
                    p2 = torch.Tensor([p[2], p[3], 1]).view(3, 1)
                    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
                    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
                    pos = torch.cat((p1_n, p2_n), 1).cuda()
                    t.pos = pos.view(1, -1)

            if self.motion_model:
                for t in self.tracks:
                    if t.last_pos.nelement() > 0:
                        p = t.last_pos[0]
                        p1 = torch.Tensor([p[0], p[1], 1]).view(3, 1)
                        p2 = torch.Tensor([p[2], p[3], 1]).view(3, 1)

                        p1_n = torch.mm(warp_matrix, p1).view(1, 2)
                        p2_n = torch.mm(warp_matrix, p2).view(1, 2)
                        pos = torch.cat((p1_n, p2_n), 1).cuda()

                        t.last_pos = pos.view(1, -1)

    def motion(self):
        """Applies a simple linear motion model that only consideres the positions at t-1 and t-2."""
        for t in self.tracks:
            # last_pos = t.pos.clone()
            # t.last_pos = last_pos
            # if t.last_pos.nelement() > 0:
            # extract center coordinates of last pos

            x1l = t.last_pos[0, 0]
            y1l = t.last_pos[0, 1]
            x2l = t.last_pos[0, 2]
            y2l = t.last_pos[0, 3]
            cxl = (x2l + x1l) / 2
            cyl = (y2l + y1l) / 2

            # extract coordinates of current pos
            x1p = t.pos[0, 0]
            y1p = t.pos[0, 1]
            x2p = t.pos[0, 2]
            y2p = t.pos[0, 3]
            cxp = (x2p + x1p) / 2
            cyp = (y2p + y1p) / 2
            wp = x2p - x1p
            hp = y2p - y1p

            # v = cp - cl, x_new = v + cp = 2cp - cl
            cxp_new = 2 * cxp - cxl
            cyp_new = 2 * cyp - cyl

            t.pos[0, 0] = cxp_new - wp / 2
            t.pos[0, 1] = cyp_new - hp / 2
            t.pos[0, 2] = cxp_new + wp / 2
            t.pos[0, 3] = cyp_new + hp / 2

            t.last_v = torch.Tensor([cxp - cxl, cyp - cyl]).cuda()

        if self.do_reid:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    # extract coordinates of current pos
                    x1p = t.pos[0, 0]
                    y1p = t.pos[0, 1]
                    x2p = t.pos[0, 2]
                    y2p = t.pos[0, 3]
                    cxp = (x2p + x1p) / 2
                    cyp = (y2p + y1p) / 2
                    wp = x2p - x1p
                    hp = y2p - y1p

                    cxp_new = cxp + t.last_v[0]
                    cyp_new = cyp + t.last_v[1]

                    t.pos[0, 0] = cxp_new - wp / 2
                    t.pos[0, 1] = cyp_new - hp / 2
                    t.pos[0, 2] = cxp_new + wp / 2
                    t.pos[0, 3] = cyp_new + hp / 2

    def step_pub(self, blob):
        """This function should be called every timestep to perform tracking with a blob = current frame
        containing the image information.
        """

        # print("I am grere !!!!!!!!!!!!!!!!")
        for t in self.tracks:
            t.last_pos = t.pos.clone()

        ###########################
        # Look for new detections #
        ###########################
        self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
        if self.public_detections:
            dets = blob['dets']
            if len(dets) > 0:
                dets = torch.cat(dets, 0)[:, :4]
                _, scores, bbox_pred, rois = self.obj_detect.test_rois(dets)
            else:
                rois = torch.zeros(0).cuda()
        else:
            _, scores, bbox_pred, rois = self.obj_detect.detect()

        if rois.nelement() > 0:
            boxes = bbox_transform_inv(rois, bbox_pred)
            boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

            # Filter out tracks that have too low person score
            # print(scores.shape)
            if scores.shape[1] > 2:
                # print("yoyoyo")
                boxes = boxes[:, 15 * 4:(15 + 1) * 4]
                scores = scores[:, 15]
            else:
                boxes = boxes[:, self.cl * 4:(self.cl + 1) * 4]
                scores = scores[:, self.cl]
            # scores = scores[:, self.cl]
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            # boxes = boxes[inds]
            det_pos = boxes[inds]
            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()

        ##################
        # Predict tracks #
        ##################
        num_tracks = 0
        nms_inp_reg = torch.zeros(0).cuda()
        if len(self.tracks):
            # align
            if self.do_align:
                # print("doing align")
                self.align(blob)
            # apply motion model
            if self.motion_model:
                self.motion()
            # regress
            person_scores = self.regress_tracks(blob)

            if len(self.tracks):

                # create nms input
                # new_features = self.get_appearances(blob)

                # nms here if tracks overlap
                nms_inp_reg = torch.cat((self.get_pos(), person_scores.add_(3).view(-1, 1)), 1)
                keep = nms(nms_inp_reg, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i]
                                         for i in list(range(len(self.tracks)))
                                         if i not in keep])

                if keep.nelement() > 0:
                    nms_inp_reg = torch.cat(
                        (self.get_pos(), torch.ones(self.get_pos().size(0)).add_(3).view(-1, 1).cuda()), 1)
                    new_features = self.get_appearances(blob)

                    self.add_features(new_features)
                    num_tracks = nms_inp_reg.size(0)
                else:
                    nms_inp_reg = torch.zeros(0).cuda()
                    num_tracks = 0

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            nms_inp_det = torch.cat((det_pos, det_scores.view(-1, 1)), 1)
        else:
            nms_inp_det = torch.zeros(0).cuda()
        if nms_inp_det.nelement() > 0:
            keep = nms(nms_inp_det, self.detection_nms_thresh)
            nms_inp_det = nms_inp_det[keep]
            # check with every track in a single run (problem if tracks delete each other)
            for i in range(num_tracks):
                nms_inp = torch.cat((nms_inp_reg[i].view(1, -1), nms_inp_det), 0)
                keep = nms(nms_inp, self.detection_nms_thresh)
                keep = keep[torch.ge(keep, 1)]
                if keep.nelement() == 0:
                    nms_inp_det = nms_inp_det.new(0)
                    break
                nms_inp_det = nms_inp[keep]

        if nms_inp_det.nelement() > 0:
            new_det_pos = nms_inp_det[:, :4]
            new_det_scores = nms_inp_det[:, 4]

            # try to redientify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)


        for t in self.tracks:
            track_ind = int(t.id)
            if track_ind not in self.results.keys():
                self.results[track_ind] = {}
            pos = t.pos[0] / blob['im_info'][0][2]
            sc = t.score
            self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc])])

        self.im_index += 1
        self.last_image = blob['data'][0][0]

        self.clear_inactive()

    def step_pub_reid(self, blob):
        """This function should be called every timestep to perform tracking with a blob = current frame
        containing the image information.
        """

        for t in self.tracks:
            t.last_pos = t.pos.clone()

        ###########################
        # Look for new detections #
        ###########################
        self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
        if self.public_detections:
            dets = blob['dets']
            if len(dets) > 0:
                dets = torch.cat(dets, 0)[:, :4]
                _, scores, bbox_pred, rois = self.obj_detect.test_rois(dets)
            else:
                rois = torch.zeros(0).cuda()
        else:
            _, scores, bbox_pred, rois = self.obj_detect.detect()

        if rois.nelement() > 0:
            boxes = bbox_transform_inv(rois, bbox_pred)
            boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

            if scores.shape[1] > 2:
                boxes = boxes[:, 15 * 4:(15 + 1) * 4]
                scores = scores[:, 15]
            else:
                boxes = boxes[:, self.cl * 4:(self.cl + 1) * 4]
                scores = scores[:, self.cl]
            # scores = scores[:, self.cl]
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            det_pos = boxes[inds]
            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()

        ##################
        # Predict tracks #
        ##################
        num_tracks = 0
        nms_inp_reg = torch.zeros(0).cuda()
        if len(self.tracks):
            # align
            if self.do_align:
                self.align(blob)
            # apply motion model
            if self.motion_model:
                self.motion()
            # regress
            person_scores = self.regress_tracks(blob)

            if len(self.tracks):
                # nms here if tracks overlap
                nms_inp_reg = torch.cat((self.get_pos(), person_scores.add_(3).view(-1, 1)), 1)
                keep = nms(nms_inp_reg, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i]
                                         for i in list(range(len(self.tracks)))
                                         if i not in keep])

                if keep.nelement() > 0:
                    nms_inp_reg = torch.cat(
                        (self.get_pos(), torch.ones(self.get_pos().size(0)).add_(3).view(-1, 1).cuda()), 1)
                    #todo size bug alert
                    new_features = self.get_appearances_new(self.get_pos())

                    self.add_features(new_features)
                    num_tracks = nms_inp_reg.size(0)
                else:
                    nms_inp_reg = torch.zeros(0).cuda()
                    num_tracks = 0

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            nms_inp_det = torch.cat((det_pos, det_scores.view(-1, 1)), 1)
        else:
            nms_inp_det = torch.zeros(0).cuda()
        if nms_inp_det.nelement() > 0:
            keep = nms(nms_inp_det, self.detection_nms_thresh)
            nms_inp_det = nms_inp_det[keep]
            # check with every track in a single run (problem if tracks delete each other)
            for i in range(num_tracks):
                nms_inp = torch.cat((nms_inp_reg[i].view(1, -1), nms_inp_det), 0)
                keep = nms(nms_inp, self.detection_nms_thresh)
                keep = keep[torch.ge(keep, 1)]
                if keep.nelement() == 0:
                    nms_inp_det = nms_inp_det.new(0)
                    break
                nms_inp_det = nms_inp[keep]

        if nms_inp_det.nelement() > 0:
            new_det_pos = nms_inp_det[:, :4]
            new_det_scores = nms_inp_det[:, 4]

            # try to redientify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid_new(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            track_ind = int(t.id)
            if track_ind not in self.results.keys():
                self.results[track_ind] = {}
            pos = t.pos[0] / blob['im_info'][0][2]
            sc = t.score
            self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc])])

        self.im_index += 1
        self.last_image = blob['data'][0][0]

        self.clear_inactive()

    def step_private_dets(self, blob):
        """This function should be called every timestep to perform tracking with a blob = current frame
        containing the image information.
        """
        for t in self.tracks:
            t.last_pos = t.pos.clone()

        ###########################
        # Look for new detections #
        ###########################
        self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])

        num_boxes = torch.LongTensor(1).cuda()
        gt_boxes = torch.FloatTensor(1).cuda()
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()
        rois, scores, bbox_pred, \
        _, _, _, _, _ = self.obj_detect(self.obj_detect.im_data, self.obj_detect.im_info, gt_boxes, num_boxes)

        scores = scores.data
        boxes = rois.data[0, :, 1:5]

        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).cuda() \
                     + torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).cuda()
        box_deltas = box_deltas.view(-1, 4 * self.obj_detect.n_classes)

        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, blob['im_info'][0][:2])

        pred_boxes /= blob['im_info'][0][2]
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        if scores.shape[1] > 2:
            inds = torch.nonzero(scores[:, 15] > 0.5).view(-1)
        else:
            inds = torch.nonzero(scores[:, 1] > 0.5).view(-1)
        # if there is det
        if inds.numel() > 0:
            if scores.shape[1] > 2:
                cls_scores = scores[:, 15][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, 15 * 4:(15 + 1) * 4]

            else:
                cls_scores = scores[:, 1][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, self.cl * 4:(self.cl + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            cls_scores = cls_scores[order]

            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep.view(-1).long()].detach().cpu().numpy()
            cls_scores = cls_scores[keep.view(-1).long()]

            # Limit to max_per_image detections *over all classes*
            max_per_image = 100
            if max_per_image > 0:
                image_scores = cls_dets[:, -1]
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    keep = np.where(cls_dets[:, -1] >= image_thresh)[0]
                    cls_dets = cls_dets[keep, :]
                    cls_scores = cls_scores[keep]

            det_pos = torch.from_numpy(cls_dets[:, :-1] * blob['im_info'][0][2].item()).cuda()
            det_scores = cls_scores
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()

        ##################
        # Predict tracks #
        ##################
        num_tracks = 0
        nms_inp_reg = torch.zeros(0).cuda()
        if len(self.tracks):
            # align
            if self.do_align:
                self.align(blob)
            # apply motion model
            if self.motion_model:
                self.motion()
            # regress
            person_scores = self.regress_tracks(blob)

            if len(self.tracks):
                # nms here if tracks overlap
                nms_inp_reg = torch.cat((self.get_pos(), person_scores.add_(3).view(-1, 1)), 1)
                keep = nms(nms_inp_reg, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i]
                                         for i in list(range(len(self.tracks)))
                                         if i not in keep])

                if keep.nelement() > 0:
                    nms_inp_reg = torch.cat(
                        (self.get_pos(), torch.ones(self.get_pos().size(0)).add_(3).view(-1, 1).cuda()), 1)
                    new_features = self.get_appearances(blob)

                    self.add_features(new_features)
                    num_tracks = nms_inp_reg.size(0)
                else:
                    nms_inp_reg = torch.zeros(0).cuda()
                    num_tracks = 0

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            nms_inp_det = torch.cat((det_pos, det_scores.view(-1, 1)), 1)
        else:
            nms_inp_det = torch.zeros(0).cuda()
        if nms_inp_det.nelement() > 0:
            keep = nms(nms_inp_det, self.detection_nms_thresh)
            nms_inp_det = nms_inp_det[keep]
            # check with every track in a single run (problem if tracks delete each other)
            for i in range(num_tracks):
                nms_inp = torch.cat((nms_inp_reg[i].view(1, -1), nms_inp_det), 0)
                keep = nms(nms_inp, self.detection_nms_thresh)
                keep = keep[torch.ge(keep, 1)]
                if keep.nelement() == 0:
                    nms_inp_det = nms_inp_det.new(0)
                    break
                nms_inp_det = nms_inp[keep]

        if nms_inp_det.nelement() > 0:
            new_det_pos = nms_inp_det[:, :4]
            new_det_scores = nms_inp_det[:, 4]

            # try to redientify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            track_ind = int(t.id)
            if track_ind not in self.results.keys():
                self.results[track_ind] = {}
            pos = t.pos[0] / blob['im_info'][0][2]
            sc = t.score
            self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc])])

        self.im_index += 1
        self.last_image = blob['data'][0][0]

        self.clear_inactive()

    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = torch.Tensor([])
        self.last_v = torch.Tensor([])
        self.gt_id = None

    def is_to_purge(self):
        """Tests if the object has been too long inactive and is to remove."""
        self.count_inactive += 1
        self.last_pos = torch.Tensor([])
        if self.count_inactive > self.inactive_patience:
            return True
        else:
            return False

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), 0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def test_features_new(self, test_features, objt_detector):
        """Compares test_features to features of this Track object"""

        if len(self.features) > 1:

            # for f in list(self.features):
                # print(f.shape)
            features = torch.cat(list(self.features), 0)
        else:
            features = self.features[0]

        real_features = objt_detector.reid_branch(features.detach())
        features = real_features.mean(0, keepdim=True)
        dist = 0.5*(1.0-F.cosine_similarity(features, test_features).unsqueeze(0))

        return dist