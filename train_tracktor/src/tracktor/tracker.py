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

from collections import deque
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from src.noise_utils import *
from .utils import bbox_overlaps, bbox_transform_inv, clip_boxes
from src.box_utils import make_single_matrix_torchV2_fast_tracktor
from src.loss_utils import colSoftMax, rowSoftMax, falsePositivePerFrame, missedObjectPerframe, \
    missedMatchErrorV3_tracktor, deepMOTPperFrame, missedMatchErrorV3_tracktor_reid


class Tracker():
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, dnn, optimizer, tracker_cfg):
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
        self.DHN = dnn
        self.optimizer = optimizer
        self.prev_asso = dict()
        self.iterations = 0
        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']
        self.evaluation_mode = tracker_cfg['evaluation_mode']
        self.reset()
        self.old_loss = 100
        self.save_freq = 20
        self.print_freq = 20
        self.mota_writer = None
        self.motp_writer = None
        self.clasf_writer = None

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        torch.cuda.empty_cache()


        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0
            self.last_image = None
            self.prev_asso = dict()

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features, gtids_=None):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            if gtids_ is not None:
                self.tracks.append(Track(new_det_pos[i].view(1,-1), new_det_scores[i], self.track_num + i, new_det_features[i].view(1,-1),
                                                                        self.inactive_patience, self.max_features_num, gtids_[i]))
            else:
                self.tracks.append(Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
                                         new_det_features[i].view(1, -1),
                                         self.inactive_patience, self.max_features_num))
        self.track_num += num_new

    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores."""
        old_pos = self.get_pos().clone()

        # regress
        _, scores, bbox_pred, rois = self.obj_detect.test_rois(old_pos)
        boxes = bbox_transform_inv(rois, bbox_pred)
        boxes = clip_boxes(boxes, blob['im_info'][0][:2])
        pos = boxes[:, self.cl*4:(self.cl+1)*4]
        scores = scores[:, self.cl]

        return scores, pos

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos

    def update_association(self, asso):
        """Get the association of all active tracks."""
        for t in self.tracks:
            asso[t.gt_id] = t.id
        return asso

    def get_pos_asso(self, asso):
        """Get the positions and associations of all active tracks."""
        collect_pos = list()
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
            asso[self.tracks[0].gt_id] = self.tracks[0].id
        elif len(self.tracks) > 1:
            for t in self.tracks:
                collect_pos.append(t.pos)
                asso[t.gt_id] = t.id
            pos = torch.cat(collect_pos, 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos, asso

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
            features = torch.cat([t.features for t in self.inactive_tracks],0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores, associated_gt_ids):
        with torch.set_grad_enabled(False):
            """Tries to ReID inactive tracks with provided detections."""
            new_det_features = self.reid_network.test_rois(blob['app_data'][0], new_det_pos / blob['im_info'][0][2]).detach()
            if len(self.inactive_tracks) >= 1 and self.do_reid:
                # calculate appearance distances
                dist_mat = []
                pos = []
                for t in self.inactive_tracks: # for each inactive track, calculate its appearance distance with all new dets
                    # print('i am here')
                    # for feat in new_det_features:
                    # 	print(t.test_features(feat.view(1,-1)).shape)
                    dist_mat.append(torch.cat([t.test_features(feat.view(1,-1)) for feat in new_det_features], 1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:  # for all inactive tracks
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos,0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]

                # calculate IoU distances between inactive track and new dets
                iou = bbox_overlaps(pos, new_det_pos)
                iou_mask = torch.ge(iou, self.reid_iou_threshold)
                iou_neg_mask = ~iou_mask
                # make all impossible assignemnts to the same add big value
                dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float()*1000
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r,c in zip(row_ind, col_ind):
                    if dist_mat[r,c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.last_v = torch.Tensor([])
                        t.pos = new_det_pos[c].view(1,-1)
                        t.add_features(new_det_features[c].view(1,-1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    associated_gt_ids = associated_gt_ids[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()
                    associated_gt_ids = torch.zeros(0).cuda()
        return new_det_pos, new_det_scores, new_det_features, associated_gt_ids

    def clear_inactive(self):
        """Checks if inactive tracks should be removed."""
        to_remove = []
        for t in self.inactive_tracks:
            if t.is_to_purge():
                to_remove.append(t)
        for t in to_remove:
            self.inactive_tracks.remove(t)

    def get_appearances_new(self, pos):
        """Uses the siamese CNN to get the features for all active tracks."""
        with torch.no_grad():
            new_features = self.obj_detect.extract_features(pos).detach()
        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1,-1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = self.last_image.cpu().numpy()
            im2 = blob['data'][0][0].cpu().numpy()
            im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
            im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
            sz = im1.shape
            warp_mode = self.warp_mode
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            #number_of_iterations = 5000
            number_of_iterations = self.number_of_iterations
            termination_eps = self.termination_eps
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
            (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
            warp_matrix = torch.from_numpy(warp_matrix)
            pos = []
            for t in self.tracks:
                p = t.pos[0]
                p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
                p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

                p1_n = torch.mm(warp_matrix, p1).view(1,2)
                p2_n = torch.mm(warp_matrix, p2).view(1,2)
                pos = torch.cat((p1_n, p2_n), 1).cuda()

                t.pos = pos.view(1,-1)

            if self.do_reid:
                for t in self.inactive_tracks:
                    p = t.pos[0]
                    p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
                    p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)
                    p1_n = torch.mm(warp_matrix, p1).view(1,2)
                    p2_n = torch.mm(warp_matrix, p2).view(1,2)
                    pos = torch.cat((p1_n, p2_n), 1).cuda()
                    t.pos = pos.view(1,-1)

            if self.motion_model:
                for t in self.tracks:
                    if t.last_pos.nelement() > 0:
                        p = t.last_pos[0]
                        p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
                        p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

                        p1_n = torch.mm(warp_matrix, p1).view(1,2)
                        p2_n = torch.mm(warp_matrix, p2).view(1,2)
                        pos = torch.cat((p1_n, p2_n), 1).cuda()

                        t.last_pos = pos.view(1,-1)

    def motion(self):
        """Applies a simple linear motion model that only consideres the positions at t-1 and t-2."""
        for t in self.tracks:
            x1l = t.last_pos[0,0]
            y1l = t.last_pos[0,1]
            x2l = t.last_pos[0,2]
            y2l = t.last_pos[0,3]
            cxl = (x2l + x1l)/2
            cyl = (y2l + y1l)/2

            # extract coordinates of current pos
            x1p = t.pos[0,0]
            y1p = t.pos[0,1]
            x2p = t.pos[0,2]
            y2p = t.pos[0,3]
            cxp = (x2p + x1p)/2
            cyp = (y2p + y1p)/2
            wp = x2p - x1p
            hp = y2p - y1p

            # v = cp - cl, x_new = v + cp = 2cp - cl
            cxp_new = 2*cxp - cxl
            cyp_new = 2*cyp - cyl

            t.pos[0,0] = cxp_new - wp/2
            t.pos[0,1] = cyp_new - hp/2
            t.pos[0,2] = cxp_new + wp/2
            t.pos[0,3] = cyp_new + hp/2

            t.last_v = torch.Tensor([cxp - cxl, cyp - cyl]).cuda()

        if self.do_reid:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    # extract coordinates of current pos
                    x1p = t.pos[0, 0]
                    y1p = t.pos[0, 1]
                    x2p = t.pos[0, 2]
                    y2p = t.pos[0, 3]
                    cxp = (x2p + x1p)/2
                    cyp = (y2p + y1p)/2
                    wp = x2p - x1p
                    hp = y2p - y1p

                    cxp_new = cxp + t.last_v[0]
                    cyp_new = cyp + t.last_v[1]

                    t.pos[0, 0] = cxp_new - wp/2
                    t.pos[0, 1] = cyp_new - hp/2
                    t.pos[0, 2] = cxp_new + wp/2
                    t.pos[0, 3] = cyp_new + hp/2

    def step_full_reid(self, blob, epoch, output_dir, is_start=False):
        """This function should be called every timestep to perform tracking with a blob = current frame
        containing the image information.
        """
        with torch.set_grad_enabled(True):
            if is_start:
                # first frame, init with gt
                gt_ids = list(blob['gt'].keys())
                if len(gt_ids) > 0:

                    for gtidx, bbox_ in blob['gt'].items():
                        bbox_to_crop = bbox_.cpu().numpy()[0, :]
                    if np.random.rand() <= 0.4:
                        bbox_to_crop = shift_box(bbox_to_crop, int(blob['im_info'][0, 0]), int(blob['im_info'][0, 1]))
                        blob['gt'][gtidx] = torch.from_numpy(bbox_to_crop).type(torch.float32).unsqueeze(0)
                    if np.random.rand() <= 0.4:
                        new_bbox = scale_box(bbox_to_crop, int(blob['im_info'][0, 0]), int(blob['im_info'][0, 1]))
                        blob['gt'][gtidx] = torch.from_numpy(new_bbox).type(torch.float32).unsqueeze(0)

                    new_gt_pos = clip_boxes(torch.cat(list(blob['gt'].values()), dim=0).type(torch.float32).cuda(), blob['im_info'][0][:2])
                    new_gt_scores = torch.zeros(new_gt_pos.shape[0]).type(torch.float32).cuda() + 1.0
                    with torch.set_grad_enabled(False):
                        self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
                        """Tries to ReID inactive tracks with provided detections."""

                        new_det_features = self.obj_detect.extract_features(new_gt_pos)

                    self.im_index += 1
                    self.last_image = blob['data'][0][0]

                    self.add(new_gt_pos, new_gt_scores, new_det_features, gt_ids)
                    self.prev_asso = self.update_association(self.prev_asso)
                return 0

            for t in self.tracks:
                t.last_pos = t.pos.clone()

            ##################
            # Predict tracks #
            ##################
            if len(self.tracks):

                self.align(blob)
                for repeat in range(1):

                    self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
                    new_scores, new_pos = self.regress_tracks(blob)

                    gt_ids = list(blob['gt'].keys())
                    if len(gt_ids) > 0:

                        gt_pos = clip_boxes(torch.cat(list(blob['gt'].values()), dim=0).type(torch.float32).cuda().clone(),
                                            blob["im_info"][0][:2])
                        dist_predt_gt = make_single_matrix_torchV2_fast_tracktor(gt_pos, new_pos, blob['im_info'][0][0],
                                                                                 blob['im_info'][0][1])

                        # dist appearance
                        dist_predt_gt_app = list()
                        gt_features = self.obj_detect.extract_features(gt_pos).detach()
                        gt_real_features = self.obj_detect.reid_branch(gt_features)  # todo detach or not ?

                        for t in self.tracks:
                            dist_predt_gt_app.append(t.test_features_new(gt_real_features, self.obj_detect))

                        dist_predt_gt_app = torch.cat(dist_predt_gt_app, dim=0).unsqueeze(0)
                        assert dist_predt_gt_app.shape == dist_predt_gt.shape

                        # get output from DHN, i.e. assignment matrix
                        output_track_gt = self.DHN(dist_predt_gt)

                        output_track_gt_app = self.DHN(dist_predt_gt_app)


                        # prepare for fn, fp
                        softmaxed_row = rowSoftMax(output_track_gt, scale=100.0, threshold=0.5).contiguous()
                        softmaxed_col = colSoftMax(output_track_gt, scale=100.0, threshold=0.5).contiguous()

                        softmaxed_row_app = rowSoftMax(output_track_gt_app, scale=100.0, threshold=0.5).contiguous()
                        softmaxed_col_app = colSoftMax(output_track_gt_app, scale=100.0, threshold=0.5).contiguous()

                        # false positives, false negatives
                        fn = missedObjectPerframe(softmaxed_col)
                        fp = falsePositivePerFrame(softmaxed_row)

                        fn_app = missedObjectPerframe(softmaxed_col_app)
                        fp_app = falsePositivePerFrame(softmaxed_row_app)

                        # idsw
                        hypo_ids = [t.id for t in self.tracks]
                        mm, mm_app, motp_mask, self.prev_asso, self.tracks = missedMatchErrorV3_tracktor_reid(self.prev_asso, gt_ids,
                                                                                                 hypo_ids, self.tracks,
                                                                                                 softmaxed_col, softmaxed_col_app,
                                                                                                 toUpdate=True)

                        sum_dist_app, matched_objects_app = deepMOTPperFrame(dist_predt_gt_app, motp_mask)
                        # motp
                        # sum of distances among matched objects and gts

                        amax = torch.argmax(softmaxed_col, dim=1)
                        motp_mask = torch.zeros_like(softmaxed_col).detach()
                        for batch in range(amax.shape[0]):
                            for width in range(motp_mask.shape[2]):
                                motp_mask[batch, amax[batch, width], width] = 1.0
                        motp_mask = motp_mask[0:, 0:-1, 0:]
                        sum_distance, matched_objects = deepMOTPperFrame(dist_predt_gt, motp_mask)

                        if not int(matched_objects) == 0:
                            motp = sum_distance / float(matched_objects)

                        else:
                            motp = torch.zeros(1).cuda()

                        if not int(matched_objects_app) == 0:
                            motp_app = sum_dist_app/float(matched_objects_app)
                        else:
                            motp_app = torch.zeros(1).cuda()

                        # check this
                        total_objects = float(dist_predt_gt.size(2))
                        if not total_objects == 0.0:
                            mota = (fn + fp) / total_objects
                            mota_app = (fn_app + fp_app + 2.0*mm_app) / total_objects
                        else:
                            mota = torch.zeros(1).cuda()
                            mota_app = torch.zeros(1).cuda()

                        loss = 5.0 * motp + mota + 5.0*motp_app + mota_app


                    else:
                        loss = torch.zeros(1).cuda()
                    if loss.item() > 0.0:
                        self.obj_detect.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                self.iterations += 1
                # save best model #
                if (self.iterations + 1) % self.save_freq == 0 and self.old_loss > loss.item() and loss.item()>0.0:
                    self.old_loss = float(loss.item())
                    print("best model is saved into:", output_dir + "/" +
                          "best_model_" + str(epoch) + ".pth")

                    torch.save(self.obj_detect.state_dict(),
                               output_dir + "/" + "best_model_" + str(epoch) + ".pth")

                    torch.save({'model_weight': self.obj_detect.state_dict(),
                                'optimizer':  self.optimizer.state_dict(),
                                'loss': self.old_loss,
                                'iterations': self.iterations},
                               output_dir + "/" + "best_model_" + str(epoch) + ".pth.tar")

                # print results #
                if (self.iterations + 1) % self.print_freq == 0 and loss.item()>0.0:
                    print('Epoch: [{}] Iterations: [{}]\tLoss {:.4f}'.format(epoch, self.iterations, float(loss.item())))

                    self.mota_writer.add_scalar('Loss', mota.item(), self.iterations)
                    self.motp_writer.add_scalar('Loss', motp.item(), self.iterations)
                    self.mota_writer.add_scalar('ReidLoss', mota_app.item(), self.iterations)
                    self.motp_writer.add_scalar('ReidLoss', motp_app.item(), self.iterations)
                    # save model #
                    if (self.iterations + 1) % (self.save_freq*10) == 0:
                        print("model is saved into:", output_dir + "/" + "model_" + str(epoch) + ".pth")

                        torch.save(self.obj_detect.state_dict(), output_dir + "/" + "model_" + str(epoch) + ".pth")
                if loss.item() > 0:
                    # update features (only matched objects)
                    find_matched = torch.sum(motp_mask[0], dim=1).float()  # of shape [motp_mask.shape[1]]
                    find_matched_bool = torch.ge(find_matched, 1)
                    self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])

                    toupdate_feat = self.get_appearances_new(new_pos)  # of shape [num_pos, feature_size]

                    assert find_matched_bool.shape[0] == new_pos.shape[0]

                    for i in range(len(self.tracks) - 1, -1, -1):
                        t = self.tracks[i]
                        t.score = new_scores[i].detach()  # update scores
                        t.pos = new_pos.detach()[i].view(1, -1)  # update pos

                        # update features
                        if find_matched_bool[i].item() >= 1:
                            t.add_features(toupdate_feat[i, :].unsqueeze(0))

            # simpler birth and death process #
            if len(self.tracks) == 0 and len(blob['gt'].keys()) != 0:  # tracks all dead but have new gt_boxes
                to_birth = list(range(len(blob['gt'].keys())))
                to_die = list()
            elif len(self.tracks) != 0 and len(blob['gt'].keys()) == 0:  # have tracks but no gt_boxes
                to_birth = list()
                to_die = list(range(len(self.tracks)))
            elif len(self.tracks) == 0 and len(blob['gt'].keys()) == 0:  # no tracks and no gt_boxes
                to_birth = list()
                to_die = list()
            else:

                motp_mask = motp_mask.detach().cpu().numpy()
                # ids to die
                sum_row = np.sum(motp_mask, axis=2)
                to_die = np.where(sum_row == 0.0)[0].tolist()

                # ids to birth
                sum_col = np.sum(motp_mask, axis=1)[0]
                to_birth = np.where(sum_col == 0.0)[0].tolist()

            if len(to_birth):
                det_boxes = clip_boxes(torch.cat(list(blob['gt'].values()), dim=0).type(torch.float32).cuda().clone(),
                                       blob["im_info"][0][:2])

                gt_ids = torch.tensor(list(blob['gt'].keys())).type(torch.int32)

                # new_gt_scores = torch.cat(list(blob['vis'].values()), dim=0).type(torch.float32).cuda()
                new_gt_scores = torch.zeros(det_boxes.shape[0]).type(torch.float32).cuda() + 1.0
                with torch.set_grad_enabled(False):
                    self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
                    """Tries to ReID inactive tracks with provided detections."""
                    new_det_features = self.obj_detect.extract_features(det_boxes[to_birth])

                self.add(det_boxes[to_birth], new_gt_scores[to_birth], new_det_features, gt_ids[to_birth].tolist())

            # put to death
            if len(to_die) > 0:
                to_die = sorted(to_die, reverse=True)

            for idx in to_die:
                t = self.tracks[idx]
                self.tracks_to_inactive([t])

            self.im_index += 1
            self.last_image = blob['data'][0][0]
            self.prev_asso = self.update_association(self.prev_asso)
            self.clear_inactive()

    def reorganize_byid(self, predict_boxes, gt_boxes, gt_ids):
        new_predicted = list()
        new_gt_boxes = list()
        for i, t in enumerate(self.tracks):
            t_id = t.gt_id
            if t_id in gt_ids:
                new_predicted.append(predict_boxes[i,:].unsqueeze(0))
                new_gt_boxes.append(gt_boxes[gt_ids.index(t_id), :].unsqueeze(0))
            else:
                pass
        new_predicted = torch.cat(new_predicted, dim=0)
        new_gt_boxes = torch.cat(new_gt_boxes, dim=0)
        return new_gt_boxes, new_predicted



class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, gtid=None):
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
        if gtid is not None:
            self.gt_id = gtid
        else:
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

    # def test_features(self, test_features):
    #     """Compares test_features to features of this Track object"""
    #     if len(self.features) > 1:
    #         features = torch.cat(list(self.features), 0)
    #     else:
    #         features = self.features[0]
    #     features = features.mean(0, keepdim=True)
    #     dist = F.pairwise_distance(features, test_features, keepdim=True)
    #     return dist

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
