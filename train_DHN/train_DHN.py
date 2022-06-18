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
import torch.optim as optim
from real_dataset import RealData
import torch
import numpy as np
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter
from utils import adjust_learning_rate, eval_acc
# from loss.DICE import soft_dice_loss
from binaryFocalloss import weighted_binary_focal_entropy
import shutil
import os
import argparse
from os.path import realpath, dirname


def main(args):
    # set all seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    if not os.path.exists(os.path.join(args.save_path, args.save_name)):
        os.makedirs(os.path.join(args.save_path, args.save_name))

    old_lr = 0.0003


    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_model_name=""):
        torch.save(state, filename)
        torch.save(state['state_dict'], filename[:-4])
        if is_best:
            shutil.copyfile(filename, best_model_name)
            shutil.copyfile(filename[:-4], best_model_name[:-4])


    # model #
    model = Munkrs(element_dim=args.element_dim, hidden_dim=args.hidden_dim, target_size=args.target_size,
                   bidirectional=args.bidirectional, minibatch=args.batch_size, is_cuda=args.is_cuda, is_train=True)
    model = model.train()
    if args.is_cuda:
        model = model.cuda()

    # optimizer #
    optimizer = optim.RMSprop(model.parameters(), lr=old_lr)

    # load and finetune model #
    starting_epoch = 0

    load_path = None
    while os.path.exists(os.path.join(args.save_path, args.save_name) + "/DHN_" + str(starting_epoch+1) + "_best.pth.tar"):
        load_path = os.path.join(args.save_path, args.save_name) + "/DHN_" + str(starting_epoch+1) + "_best.pth.tar"
        starting_epoch += 1

    if load_path is not None:
        model_params = torch.load(load_path)
        print('loading model from: ')
        print(load_path)
        print('with acc:')
        print(model_params['best_prec1'])

        old_acc = model_params['best_prec1']
        print('at iterations:')

        starting_iterations = model_params['iters']
        print(starting_iterations)
        model.load_state_dict(model_params['state_dict'])
        optimizer.load_state_dict(model_params['optimizer'])


    else:
        if os.path.exists(args.logs_path):
            print("remove old logs")
            shutil.rmtree(args.logs_path+'train', ignore_errors=True)
            shutil.rmtree(args.logs_path + 'test', ignore_errors=True)
        starting_epoch = 0
        starting_iterations = 0
        old_acc = 0.0


    # TensorboardX logs #

    train_writer = SummaryWriter(os.path.join(args.logs_path, 'train'))
    val_writer = SummaryWriter(os.path.join(args.logs_path, 'test'))


    # data loaders #
    train_dataset = RealData(args.data_path, train=True)
    val_dataset = RealData(args.data_path, train=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_dataloader =torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    print('val length in #batches: ', len(val_dataloader.dataset))
    print('train length in #batches: ', len(train_dataloader.dataset))

    is_best = False
    val_loss = None

    iteration = 0
    iteration += starting_iterations
    if iteration > 0 and iteration % len(train_dataloader.dataset) == 0:
        starting_epoch += 1
    for epoch in range(max(0, starting_epoch), args.epochs):

        for Dt, target in train_dataloader:

            model = model.train()
            Dt = Dt.squeeze(0)
            target = target.squeeze(0)
            if args.is_cuda:
                Dt = Dt.cuda()
                target = target.cuda()

            # after each sequence/matrix Dt, we init new hidden states
            model.hidden_row = model.init_hidden(Dt.size(0))
            model.hidden_col = model.init_hidden(Dt.size(0))

            # input to model
            tag_scores = model(Dt)
            # num_positive = how many labels = 1
            num_positive = target.detach().clone().view(target.size(0), -1).sum(dim=1).unsqueeze(1)
            weight2negative = num_positive.float()/(target.size(1)*target.size(2))
            # case all zeros, then weight2negative = 1.0
            weight2negative.masked_fill_((weight2negative == 0.0), 10)  # 10 is just a symbolic value representing 1.0
            # case all ones, then weight2negative = 0.0
            weight2negative.masked_fill_((weight2negative == 1.0), 0.0)
            # change all fake values 10 to their desired value 1.0
            weight2negative.masked_fill_((weight2negative == 10), 1.0)
            weight = torch.cat([weight2negative, 1.0 - weight2negative], dim=1)
            weight = weight.view(-1, 2, 1, 1).contiguous()
            if args.is_cuda:
                weight = weight.cuda()

            loss = 10.0 * weighted_binary_focal_entropy(tag_scores, target.float(), weights=weight)

            # clean gradients & back propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # adjust learning weight
            old_lr = adjust_learning_rate(optimizer, iteration, old_lr)

            # show loss
            train_writer.add_scalar('Loss', loss.item(), iteration)
            if val_loss is None:
                val_loss = loss.item()
            val_writer.add_scalar('Loss', val_loss, iteration)

            if iteration % args.print_train == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.4f}'.format(epoch, iteration % len(train_dataloader.dataset),
                                                               len(train_dataloader.dataset), loss.item()))

            if iteration % args.print_test == 0:
                model = model.eval()
                test_loss = []
                acc = []
                test_j = []
                test_p = []
                test_r = []
                # val = random.sample(valset, 50)
                for test_num, (data, target) in enumerate(val_dataloader):
                    data = data.squeeze(0)
                    target = target.squeeze(0)
                    if test_num == 50:
                        break
                    if args.is_cuda:
                        data = data.cuda()
                        target = target.cuda()
                    # after each sequence/matrix Dt, we init new hidden states
                    model.hidden_row = model.init_hidden(data.size(0))
                    model.hidden_col = model.init_hidden(data.size(0))

                    # input to model
                    tag_scores = model(data)

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
                    # print weight
                    if args.is_cuda:
                        weight = weight.cuda()

                    loss = 10.0 * weighted_binary_focal_entropy(tag_scores, target.float(), weights=weight)

                    test_loss.append(float(loss.item()))
                    # scores = F.sigmoid(tag_scores)
                    predicted, curr_acc = eval_acc(tag_scores.float().detach(), target.float().detach(), weight.detach())
                    acc.append(curr_acc)

                    # calculate J value
                    tp = torch.sum((predicted == target.float().detach())[target.data == 1.0]).double()
                    fp = torch.sum((predicted != target.float().detach())[predicted.data == 1.0]).double()
                    fn = torch.sum((predicted != target.float().detach())[predicted.data == 0.0]).double()

                    p = tp / (tp + fp + 1e-9)
                    r = tp / (tp + fn + 1e-9)
                    test_p.append(p.item())
                    test_r.append(r.item())

                print('Epoch: [{}][{}/{}]\tLoss {:.4f}\tweighted Accuracy {:.2f} %'.format(epoch, iteration % len(train_dataloader.dataset),
                                                                                           len(train_dataloader.dataset),
                                                                                            np.mean(np.array(test_loss)),
                                                                                            100.0*np.mean(np.array(acc))))

                print('P {:.2f}% \t R {:.2f}%'.format(100.0*np.mean(np.array(test_p)), 100.0*np.mean(np.array(test_r))))

                # show loss and accuracy
                val_loss = np.mean(np.array(test_loss))
                val_writer.add_scalar('Weighted Accuracy', np.mean(np.array(acc)), iteration)

                val_writer.add_scalar('recall', np.mean(np.array(test_r)), iteration)

                val_writer.add_scalar('precision', np.mean(np.array(test_p)), iteration)
                val_writer.add_scalar('Loss', val_loss, iteration)

                if old_acc < np.mean(np.array(acc)):
                    old_acc = np.mean(np.array(acc)) + 0.0
                    is_best = True

                # save checkpoints
                save_checkpoint({
                    'epoch': epoch + 1,
                    'iters': iteration,
                    'state_dict': model.state_dict(),
                    'best_prec1': np.mean(np.array(acc)),
                    'optimizer': optimizer.state_dict(),
                }, is_best, os.path.join(args.save_path, args.save_name) + "/DHN_" + str(epoch+1) + "_.pth.tar",
                best_model_name=os.path.join(args.save_path, args.save_name) + "/DHN_" + str(epoch+1) + "_best.pth.tar")

                is_best = False
            iteration += 1

if __name__ == '__main__':
    # parameters #

    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='PyTorch DeepMOT train')

    # data configs
    parser.add_argument('--data_path', dest='data_path', default=os.path.join(curr_path, 'DHN_data/'), help='dataset root path')
    parser.add_argument('--is_cuda', action='store_true', help="use GPU if set.")

    # BiRNN configs
    parser.add_argument('--element_dim', dest='element_dim', default=1, type=int, help='element_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int, help='hidden_dim')
    parser.add_argument('--target_size', dest='target_size', default=1, type=int, help='target_size')
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--bidirectional', action='store_true', help="Bi-RNN if set.")

    # train configs
    parser.add_argument('-b', dest='batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', dest='epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('--print_test', dest='print_test', default=20, type=int, help='test frequency')
    parser.add_argument('--print_train', dest='print_train', default=10, type=int, help='training print frequency')
    parser.add_argument('--logs_path', dest='logs_path', default=os.path.join(curr_path, 'log/'), help='log files path')
    parser.add_argument('--save_path', dest='save_path', default=os.path.join(curr_path, 'output/'), help='save path')
    parser.add_argument('--save_name', dest='save_name', default='DHN', help='save folder name')

    args = parser.parse_args()

    main(args)