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
from collections import defaultdict
from os import path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from cycler import cycler as cy
from scipy.interpolate import interp1d
import pandas as pd
import cv2
import copy
import csv
persons_class = ["1"]

colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']

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


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
	"""
	Parameters
	----------
	boxes: (N, 4) ndarray or tensor or variable
	query_boxes: (K, 4) ndarray or tensor or variable
	Returns
	-------
	overlaps: (N, K) overlap between boxes and query_boxes
	"""
	if isinstance(boxes, np.ndarray):
		boxes = torch.from_numpy(boxes)
		query_boxes = torch.from_numpy(query_boxes)
		out_fn = lambda x: x.numpy() # If input is ndarray, turn the overlaps back to ndarray when return
	else:
		out_fn = lambda x: x

	box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
			(boxes[:, 3] - boxes[:, 1] + 1)
	query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
			(query_boxes[:, 3] - query_boxes[:, 1] + 1)

	iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
	ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
	ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
	overlaps = iw * ih / ua
	return out_fn(overlaps)


def plot_img(data_root, dataset, result_path, save_path, seq_folder_name, seq_name, colorList):

	path_label = ''

	if "MOT17" in dataset:
		dataset = "MOT17Det"
		label = "MOT17Labels"
	elif "MOT16" in dataset:
		dataset = "MOT17Det"
		label = "MOT16Labels"

	if os.path.exists(data_root + dataset + "/train/" + 'MOT17-'+seq_folder_name[6:8]):
		path_data = data_root + dataset + "/train/" + 'MOT17-'+seq_folder_name[6:8] + "/"
		# path_label = data_root + label + "/train/" + seq_name + "/gt/"

	elif os.path.exists(data_root + dataset + "/test/" + 'MOT17-'+seq_folder_name[6:8]):
		path_data = data_root + dataset + "/test/" + 'MOT17-'+seq_folder_name[6:8] + "/"
	else:
		return 0

	path_res = result_path
	path_images = path_data + '/img1/'

	save_path = save_path + '/' + seq_name + '/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	else:
		return

	if 'MOT' not in seq_name:
		return

	res_raw = pd.read_csv(path_res, sep=',', header=None)
	res_raw = np.array(res_raw).astype(np.float32)
	res_raw[:, 0:6] = np.array(res_raw[:, 0:6]).astype(np.int)
	if len(path_label) > 0:
		path_gt = path_label + 'gt.txt'
		gt_raw = read_txt_gtV2(path_gt)

	N_frame = max(res_raw[:, 0])
	print('total number of frames: ', N_frame)
	# N_frame = 100

	for t in range(1, int(N_frame)):
		if os.path.exists(save_path + str(t).zfill(6) + '.jpg'):
			continue
		print('t = ' + str(t))
		img_name = path_images + str(t).zfill(6) + '.jpg'
		print(img_name)
		img = cv2.imread(img_name)
		overlay = img.copy()

		# plot gt if exists
		if len(path_label) > 0:
			if str(t) in gt_raw.keys():
				for gt_visu in gt_raw[str(t)]:
					cv2.rectangle(overlay, (int(gt_visu[1]), int(gt_visu[2])), (int(gt_visu[3]), int(gt_visu[4])),
								  (0, 255, 0), 3)
					cv2.putText(overlay, gt_visu[0], (int(gt_visu[1]), -10 + int(gt_visu[2])),
								cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

		# print(img.shape)

		row_ind = np.where(res_raw[:, 0] == t)[0]
		for i in range(0, row_ind.shape[0]):
			id = int(max(res_raw[row_ind[i], 1], 0))
			color_ind = id % len(colorList)

			# plot the line
			row_ind_line = np.where((res_raw[:, 0] > t - 50) & (res_raw[:, 0] < t + 1) & (res_raw[:, 1] == id))[0]

			# plot the rectangle
			for j in range(0, row_ind_line.shape[0], 5):
				line_xc = int(res_raw[row_ind_line[j], 2] + 0.5 * res_raw[row_ind_line[j], 4])
				line_yc = int(res_raw[row_ind_line[j], 3] + res_raw[row_ind_line[j], 5])
				bb_w = 5
				line_x1 = line_xc - bb_w
				line_y1 = line_yc - bb_w
				line_x2 = line_xc + bb_w
				line_y2 = line_yc + bb_w
				cv2.rectangle(overlay, (line_x1, line_y1), (line_x2, line_y2), colorList[color_ind], -1)

				t_past = res_raw[row_ind_line[j], 0]
				alpha = 1 - (t - t_past) / 80  # Transparency factor.
				img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
				overlay = img.copy()

		for i in range(0, row_ind.shape[0]):
			id = int(res_raw[row_ind[i], 1])
			bb_x1 = int(res_raw[row_ind[i], 2])
			bb_y1 = int(res_raw[row_ind[i], 3])
			bb_x2 = int(res_raw[row_ind[i], 2] + res_raw[row_ind[i], 4])
			bb_y2 = int(res_raw[row_ind[i], 3] + res_raw[row_ind[i], 5])
			str_tmp = str(i) + ' ' + str("{0:.2f}".format(res_raw[row_ind[i], 6]))
			color_ind = id % len(colorList)
			cv2.rectangle(overlay, (bb_x1, bb_y1), (bb_x2, bb_y2), colorList[color_ind], 3)
		# cv2.imshow('image', overlay)
		# cv2.waitKey(-1)
		save_name = save_path + str(t).zfill(6) + '.jpg'
		cv2.imwrite(save_name, overlay)


def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
	#output_dir = get_output_dir("anchor_gt_demo")
	im_paths = blobs['im_paths']
	if not name:
		im0_name = osp.basename(im_paths[0])
	else:
		im0_name = str(name)+".jpg"
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]

	im_scales = blobs['im_info'][0,2]

	tracks = tracks.data.cpu().numpy() / im_scales
	num_tracks = tracks.shape[0]

	fig, ax = plt.subplots(1,2,figsize=(12, 6))

	ax[0].imshow(im0, aspect='equal')
	ax[1].imshow(im1, aspect='equal')

	# infinte color loop
	cyl = cy('ec', colors)
	loop_cy_iter = cyl()
	styles = defaultdict(lambda : next(loop_cy_iter))

	ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

	for i,t in enumerate(tracks):
		t0 = t[0]
		t1 = t[1]
		ax[0].add_patch(
			plt.Rectangle((t0[0], t0[1]),
					  t0[2] - t0[0],
					  t0[3] - t0[1], fill=False,
					  linewidth=1.0, **styles[i])
			)
		ax[1].add_patch(
			plt.Rectangle((t1[0], t1[1]),
					  t1[2] - t1[0],
					  t1[3] - t1[1], fill=False,
					  linewidth=1.0, **styles[i])
			)

	if gt_tracks:
		for gt in gt_tracks:
			for i in range(2):
				ax[i].add_patch(
				plt.Rectangle((gt[i][0], gt[i][1]),
					  gt[i][2] - gt[i][0],
					  gt[i][3] - gt[i][1], fill=False,
					  edgecolor='blue', linewidth=1.0)
				)

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	image = None
	if output_dir:
		im_output = osp.join(output_dir,im0_name)
		plt.savefig(im_output)
	else:
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
		image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	plt.close()
	return image


def interpolate(tracks):
	interpolated = {}
	for i, track in tracks.items():
		interpolated[i] = {}
		frames = []
		x0 = []
		y0 = []
		x1 = []
		y1 = []

		for f, bb in track.items():
			frames.append(f)
			x0.append(bb[0])
			y0.append(bb[1])
			x1.append(bb[2])
			y1.append(bb[3])

		if len(frames) > 1:
			x0_inter = interp1d(frames, x0)
			y0_inter = interp1d(frames, y0)
			x1_inter = interp1d(frames, x1)
			y1_inter = interp1d(frames, y1)

			for f in range(min(frames), max(frames)+1):
				bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
				interpolated[i][f] = bb
		else:
			interpolated[i][frames[0]] = np.array([x0[0], y0[0], x1[0], y1[0]])

	return interpolated


def bbox_transform_inv(boxes, deltas):
  # Input should be both tensor or both Variable and on the same device
  if len(boxes) == 0:
    return deltas.detach() * 0

  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = deltas[:, 0::4]
  dy = deltas[:, 1::4]
  dw = deltas[:, 2::4]
  dh = deltas[:, 3::4]

  pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
  pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
  pred_w = torch.exp(dw) * widths.unsqueeze(1)
  pred_h = torch.exp(dh) * heights.unsqueeze(1)

  pred_boxes = torch.cat(
      [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                pred_ctr_y - 0.5 * pred_h,
                                pred_ctr_x + 0.5 * pred_w,
                                pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
  return pred_boxes


def clip_boxes(boxes, im_shape):
  """
  Clip boxes to image boundaries.
  boxes must be tensor or Variable, im_shape can be anything but Variable
  """

  if not hasattr(boxes, 'data'):
    boxes_ = boxes.numpy()

  boxes = boxes.view(boxes.size(0), -1, 4)
  boxes = torch.stack(
      [boxes[:, :, 0].clamp(0, im_shape[1] - 1),
       boxes[:, :, 1].clamp(0, im_shape[0] - 1),
       boxes[:, :, 2].clamp(0, im_shape[1] - 1),
       boxes[:, :, 3].clamp(0, im_shape[0] - 1)], 2).view(boxes.size(0), -1)

  return boxes
