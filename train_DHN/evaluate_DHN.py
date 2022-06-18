#==========================================================================
# This file is under License LGPL-3.0 (see details in the license file).
# This file is a part of implementation for paper:
# How To Train Your Deep Multi-Object Tracker.
# This contribution is headed by Perception research team, INRIA.
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
# created on 16th April 2020.
#==========================================================================

from DHN import Munkrs
import argparse
from os.path import realpath, dirname
import torch
from torch.utils import data
import os
import numpy as np
torch.set_grad_enabled(False)


def eval_acc(score, target, weight, args):
    """
    :param score: torch tensor, predicted score of shape [batch, H, W]
    :param target: torch tensor, ground truth value {0,1} of shape [batch, H, W]
    :param weight: torch tensor, weight for each batch for negative and positive examples of shape [batch, 2, 1, 1]
    :return: accuracy
    """
    acc = []
    predicted = torch.zeros_like(score)
    for b in range(score.size(0)):
        for h in range(score.size(1)):
            value, indice = score[b, h].max(0)
            if float(value) > args.threshold:
                predicted[b, h, int(indice)] = 1.0
    num_positive = float(target[b, :, :].sum())
    num_negative = float(target.size(1)*target.size(2) - num_positive)
    num_tp = float(((predicted[b, :, :] == target[b, :, :]) + (target[b, :, :] == 1.0)).eq(2).sum())
    num_tn = float(((predicted[b, :, :] == target[b, :, :]) + (target[b, :, :] == 0.0)).eq(2).sum())

    acc.append((num_tp * float(weight[b, 1, 0, 0]) + num_tn * float(weight[b, 0, 0, 0]))/
               (num_positive * float(weight[b, 1, 0, 0]) + num_negative * float(weight[b, 0, 0, 0])))

    return np.mean(np.array(acc))


def prepare_Data(data_pth):
    """
    :param data_pth: string that gives the data path
    :return: data list
    """
    dirs = os.listdir(data_pth)
    data = []
    for dir in dirs:
        pth = os.path.join(data_pth, dir)
        files = os.listdir(pth)
        for file in files:
            if '_m.npy' in file:
                data.append(os.path.join(pth, file))
    return data


class MatrixRealData(data.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, data_path):
        'Initialization'
        self.data_pth = data_path
        self.data = prepare_Data(data_path)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        m_pth = self.data[index]  # input matrix name
        # print m_pth
        split = m_pth.split('_')
        t_pth = '_'.join(split[:-1])+'_t.npy'
        # print m_pth
        matrix = torch.from_numpy(np.load(m_pth).astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))

        return [matrix, target]


def main(model, tst_dataloader, args):
    # calculate weights #
    acc = []
    running_corrects = 0
    running_tp = 0
    running_fp = 0
    running_fn = 0
    elements_count = 0.0
    count_lines = 0.0
    sum_many_ones = 0.0
    sum_all_zeros = 0.0
    total_matrices = 0.0
    wrong_matrices_1 = 0.0
    wrong_matrices_2 = 0.0
    for data, target in tst_dataloader:
        many_ones = 0
        not_all_zeros = 0
        curr_wrong_matrices_1 = 0
        curr_wrong_matrices_2 = 0

        if args.is_cuda:
            data = data.squeeze(0).cuda()
            target = target.squeeze(0).cuda()
        else:
            data = data.squeeze(0)
            target = target.squeeze(0)

        elements_count += data.shape[0]*data.shape[1]*data.shape[2]

        model.hidden_row = model.init_hidden(data.size(0))
        model.hidden_col = model.init_hidden(data.size(0))

        tag_scores = model(data).detach()

        # discretization #
        if args.row_wise:
            print("we are here")
            predicted = torch.zeros_like(tag_scores)
            for b in range(tag_scores.size(0)):
                for h in range(tag_scores.size(1)):
                    value, indice = tag_scores[b, h].max(0)
                    if float(value) > args.threshold:
                        predicted[b, h, int(indice)] = 1.0
        else:

            predicted = torch.zeros_like(tag_scores)
            for b in range(tag_scores.size(0)):
                for w in range(tag_scores.size(2)):
                    value, indice = tag_scores[b, :, w].max(0)
                    if float(value) > args.threshold:
                        predicted[b, int(indice), w] = 1.0


        # weighted accuracy #
        num_positive = target.data.view(target.size(0), -1).sum(dim=1).unsqueeze(1)
        # print num_positive
        weight2negative = num_positive.float() / (target.size(1) * target.size(2))
        # case all zeros
        weight2negative.masked_fill_((weight2negative == 0), 10)  # 10 is just a symbolic value representing 1.0
        # case all ones
        weight2negative.masked_fill_((weight2negative == 1), 0.0)
        weight2negative.masked_fill_((weight2negative == 10), 1.0)  # change all 100 to their true value 1.0
        weight = torch.cat([weight2negative, 1.0 - weight2negative], dim=1)
        # print weight
        weight = weight.view(-1, 2, 1, 1).contiguous()
        if args.is_cuda:
            weight = weight.cuda()
        acc.append(eval_acc(tag_scores.data, target.float().data, weight, args))

        # TP, TN, FP, FN, F1_score #
        target = target.float()
        running_corrects += torch.sum(predicted == target.data).double()
        running_tp += torch.sum((predicted == target.data)[target.data == 1]).double()
        running_fp += torch.sum((predicted != target.data)[predicted.data == 1]).double()
        running_fn += torch.sum((predicted != target.data)[predicted.data == 0]).double()

        # constraints #
        if args.row_wise:
            print("we are here")
            for b in range(tag_scores.size(0)):
                wrong_flag_1 = False
                wrong_flag_2 = False
                for w in range(tag_scores.size(2)):
                    sum_column_predict = torch.sum(predicted[b, :, w])
                    sum_column_gt = torch.sum(target[b,:,w])
                    if sum_column_predict.float().item() > 1.0:
                        many_ones += 1.0
                        wrong_flag_1 = True

                    elif sum_column_gt.float().item() == 0.0 and sum_column_predict.float().item() == 1.0:
                        not_all_zeros += 1.0
                        wrong_flag_2 = True
                    elif sum_column_gt.float().item() == 1.0 and sum_column_predict.float().item() == 0.0:
                        not_all_zeros += 1.0
                        wrong_flag_2 = True
                if wrong_flag_1:
                    curr_wrong_matrices_1 += 1.0
                elif wrong_flag_2:
                    curr_wrong_matrices_2 += 1.0
            print('curr wrong matrix rate:',
                  (float(curr_wrong_matrices_1) + float(curr_wrong_matrices_2)) / target.shape[0])
            print('curr not_all_zeros matrices: ', float(curr_wrong_matrices_2))
            print('curr many_ones matrices: ', float(curr_wrong_matrices_1))
            print('total wrong matrices: ', float(curr_wrong_matrices_1) + float(curr_wrong_matrices_2))
            print('total matrices: ', target.shape[0])
            print('curr many_ones lines: ', float(many_ones) / (target.shape[0] * target.shape[2]))
            count_lines += target.shape[0] * target.shape[2]
            total_matrices += target.shape[0]
            wrong_matrices_1 += curr_wrong_matrices_1
            wrong_matrices_2 += curr_wrong_matrices_2
            sum_all_zeros += not_all_zeros
            sum_many_ones += many_ones
            print()
        else:
            for b in range(tag_scores.size(0)):
                wrong_flag_1 = False
                wrong_flag_2 = False
                for h in range(tag_scores.size(1)):
                    sum_column_predict = torch.sum(predicted[b, h, :])
                    sum_column_gt = torch.sum(target[b, h, :])
                    if sum_column_predict.float().item() > 1.0:
                        many_ones += 1.0
                        wrong_flag_1 = True

                    elif sum_column_gt.float().item() == 0.0 and sum_column_predict.float().item() == 1.0:
                        not_all_zeros += 1.0
                        wrong_flag_2 = True
                    elif sum_column_gt.float().item() == 1.0 and sum_column_predict.float().item() == 0.0:
                        not_all_zeros += 1.0
                        wrong_flag_2 = True
                if wrong_flag_1:
                    curr_wrong_matrices_1 += 1.0
                elif wrong_flag_2:
                    curr_wrong_matrices_2 += 1.0
            print('curr wrong matrix rate:',
                  (float(curr_wrong_matrices_1) + float(curr_wrong_matrices_2)) / target.shape[0])
            print('curr not_all_zeros matrices: ', float(curr_wrong_matrices_2))
            print('curr many_ones matrices: ', float(curr_wrong_matrices_1))
            print('total wrong matrices: ', float(curr_wrong_matrices_1) + float(curr_wrong_matrices_2))
            print('total matrices: ', target.shape[0])
            print('curr many_ones lines: ', float(many_ones) / (target.shape[0] * target.shape[1]))
            count_lines += target.shape[0] * target.shape[1]
            total_matrices += target.shape[0]
            wrong_matrices_1 += curr_wrong_matrices_1
            wrong_matrices_2 += curr_wrong_matrices_2
            sum_all_zeros += not_all_zeros
            sum_many_ones += many_ones
            print()

    epoch_acc = running_corrects.double() / elements_count
    tp = running_tp.double()
    fp = running_fp.double()
    fn = running_fn.double()
    tn = 1.0*elements_count - tp - fp - fn
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)

    J = r + (tn/(tn+fp)) - 1

    epoch_f1 = 2 * p * r / (p + r + 1e-9)

    print('fn: ', fn.item())
    print('fp: ', fp.item())
    print('tp: ', tp.item())
    print('tn: ', tn.item())
    print('total elements: ', elements_count)
    print('precision: ', p.item())
    print('recall: ', r.item())
    print('Youden value:', J.item())
    print('f1_score: ', epoch_f1.item())

    print('weighted acc: ', np.mean(np.array(acc)) * 100)

    print('constraints: ')
    print('not_all_zeros wrong lines rate: ',  sum_all_zeros/count_lines)
    print('many ones wrong lines rate: ', many_ones / count_lines)
    print('total wrong matrices: ', wrong_matrices_1 +wrong_matrices_2)
    print('total matrices: ', total_matrices)
    print('many ones wrong matrices', wrong_matrices_1)
    print('many ones wrong matrices rate', wrong_matrices_1/total_matrices)
    print('not all zeros wrong matrices', wrong_matrices_2)
    print('not all zeros wrong matrices rate', wrong_matrices_2/total_matrices)


if __name__ == '__main__':
    # parameters #

    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='PyTorch DeepMOT train')

    # data configs
    parser.add_argument('--data_root', dest='data_root',
                        default='',
                        help='dataset root path')

    parser.add_argument('--is_cuda', dest='is_cuda', default=True, type=bool, help='use GPU?')

    parser.add_argument('--model_path', dest='model_path',
                        default="",
                        help='pretrained model path')

    # BiRNN configs

    parser.add_argument('--element_dim', dest='element_dim', default=1, type=int, help='element_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int, help='hidden_dim')
    parser.add_argument('--target_size', dest='target_size', default=1, type=int, help='target_size')
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--bidrectional', dest='bidrectional', default=True, type=bool, help='bidrectional')

    # constraints
    parser.add_argument('--row_wise', dest='row_wise', default=True, type=bool, help='row wise soft-max? column wise '
                                                                                     'if False')
    parser.add_argument('--threshold', dest='threshold', default=0.7, type=float, help = 'threshold for dicretization.')

    args = parser.parse_args()

    # load model #

    model = Munkrs(element_dim=args.element_dim, hidden_dim=args.hidden_dim, target_size=args.target_size,
                   biDirenction=args.bidrectional, minibatch=args.batch_size, is_cuda=args.is_cuda, is_train=False)
    model.eval()
    if args.is_cuda:
        model.cuda()

    # load data #

    tst_dataset = MatrixRealData(args.data_root)
    tst_dataloader = data.DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model.load_state_dict(torch.load(args.model_path))

    main(model, tst_dataloader, args)