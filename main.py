from __future__ import print_function

import os
import matplotlib.pyplot as plot
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from option import Options
from utils import *
from utils import CrossEntropyLabelSmooth
from tqdm import tqdm
from PIL import Image
import time
import numpy as np
import cv2

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []


def main():
    # init the args
    args = Options().parse()
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    args.plot = False
    args.dataset = 'data'
    args.model = 'deepten'
    args.batch_size = 128  # 128 #32
    args.loss = 'CrossEntropyLabelSmooth'
    args.backbone = 'resnet18'

    args.lr = 0.001  # 0.004  #0.01 #
    args.epochs = 60  # 60#20 #60
    args.lr_step = 15
    args.test_aug = False
    args.nclass = 24
    args.ohem = -2
    args.atte = False
    args.mixup = False
    args.basepath = './'

    # plot
    if args.plot:
        print('=>Enabling matplotlib for display:')
        plot.ion()
        plot.show()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader

    dataset = importlib.import_module('dataset.' + args.dataset)
    Dataloder = dataset.Dataloder
    # if args.eval:
    #    test_loader= Dataloder(args).gettestloader()
    # else:
    train_loader, test_loader = Dataloder(args).getloader()
    # init the model
    models = importlib.import_module('model.' + args.model)
    model = models.Net(args)
    # print(model)

    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'CrossEntropyLabelSmooth':
        criterion = CrossEntropyLabelSmooth(num_classes=args.nclass, ignore_index=-1)
    else:
        raise Keyerror('Not implement!')

    # criterion = FocalLoss(gamma=2, alpha=0.2, \
    #                                  num_classes=2,
    #                                  epsilon=0.1)

    criterion_center = CenterLoss_5(num_classes=2, feat_dim=512
                                    )

    optimizer = get_optimizer(args, model, False)
    if args.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model)
    # check point

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            # errlist_train = checkpoint['errlist_train']
            # errlist_val = checkpoint['errlist_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            for name, param in checkpoint['state_dict'].items():
                print(name)
        else:
            print("=> no resume checkpoint found at '{}'". \
                  format(args.resume))

    if args.eval == False:
        scheduler = LR_Scheduler(args, len(train_loader))

    def train(epoch):
        model.train()
        global best_pred, errlist_train
        train_loss, correct, total = 0, 0, 0
        # adjust_learning_rate(optimizer, args, epoch, best_pred)
        tbar = tqdm(train_loader, desc='\r')

        batch_idx_end = 0

        for batch_idx, (data, target, ids) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output, f, center_weight, softmax_out = model(data)

            # May need Ohem later.
            # Here leave it for future usage.
            if args.ohem > -1:
                if args.loss == 'CrossEntropyLoss':
                    loss = nn.CrossEntropyLoss(reduce=False)(output, target).cpu().detach().numpy()
                elif args.loss == 'CrossEntropyLabelSmooth':
                    loss = CrossEntropyLabelSmooth(num_classes=args.nclass, reduce=False)(output, target).cpu().detach().numpy()
                else:
                    raise Keyerror('Not implement!')
                loss_index = np.argsort(loss)
                targets_copy = target.clone()
                if args.nclass < 3:
                    targets_copy[loss_index[0:args.ohem]] = -1

            else:
                targets_copy = target.clone()

            loss = criterion(output, targets_copy)
            # print(output.shape)
            # print(targets_copy.shape)
            # center_loss = 0.0 * criterion_center(f, target, center_weight)

            # loss = loss + center_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.data
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            total += target.size(0)
            err = 100. - 100. * correct / total
            tbar.set_description('\rLoss: %.3f | Err: %.3f%% (%d/%d)' % \
                                 (train_loss / (batch_idx + 1), err, total - correct, total))
            train_loss_end = train_loss
            batch_idx_end = batch_idx

        print('\rLoss: %.3f | Err: %.3f%% (%d/%d)' % \
              (train_loss / (batch_idx_end + 1), err, total - correct, total))

    # Place holder
    # Test function has not been implemented. Wait for later finishing
    # Here only leave a skeleton code.
    def test(epoch):
        model.eval()
        global best_pred, errlist_train, errlist_val
        test_loss, correct, total = 0, 0, 0
        is_best = False
        tbar = tqdm(test_loader, desc='\r')

        batch_idx_end = 0

        TP = 0
        FN = 0
        FP = 0
        TN = 0

        with torch.no_grad():
            for batch_idx, (data, target, ids) in enumerate(tbar):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                placeholder = model(data)
                test_loss += 0  # criterion(output, target).data

                # pred = output.data.max(1)[1]
                #
                # is_right = pred.eq(target.data).cpu().sum()
                # correct += is_right
                # total += target.size(0)
                #
                # err = 100. - 100. * correct / total
                # tbar.set_description('Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                #                      (test_loss / (batch_idx + 1), err, total - correct, total))
                # batch_idx_end = batch_idx

        # recall_1 = TP / (TP + FN)
        # precision_1 = TP / (TP + FP)
        # recall_0 = TN / (TN + FP)
        # precision_0 = TN / (TN + FN)

    #     print(
    #         'Loss: %.3f | Err: %.3f%%, Recall_1:%.3f%%, Precision_1:%.3f%%, Recall_0:%.3f%%, Precision_0:%.3f%%, (%d/%d)' % \
    #         (
    #         test_loss / (batch_idx_end + 1), err, recall_1, precision_1, recall_0, precision_0, total - correct, total))
    #     print('TN: %d | TP: %d | FN: %d | FP: %d' % (TN, TP, FN, FP))
    #     if args.eval:
    #         print('Error rate is %.3f%%' % err)
    #         return
    #     # save checkpoint
    #     errlist_val += [err]
    #     if err < best_pred:
    #         best_pred = err
    #         is_best = True
    #     print(err)
    #     print('Best Error rate is %.3f%%' % best_pred)
    #
    #     time.sleep(10)
    #     save_checkpoint({
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'best_pred': best_pred,
    #         'errlist_train': errlist_train,
    #         'errlist_val': errlist_val,
    #     }, args=args, is_best=is_best)
    #
    #     if args.plot:
    #         plot.clf()
    #         plot.xlabel('Epoches: ')
    #         plot.ylabel('Error Rate: %')
    #         plot.plot(errlist_train, label='train')
    #         plot.plot(errlist_val, label='val')
    #         plot.legend(loc='upper left')
    #         plot.draw()
    #         plot.pause(0.001)
    #
    if args.eval:
        test(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(epoch)
        test(epoch)

    # save train_val curve to a file
    if args.plot:
        plot.clf()
        plot.xlabel('Epoches: ')
        plot.ylabel('Error Rate: %')
        plot.plot(errlist_train, label='train')
        plot.plot(errlist_val, label='val')
        plot.savefig("%s/runs/%s/%s/%s/" % (args.basepath, args.dataset, args.model, args.checkname)
                     + 'train_val.jpg')

if __name__ == "__main__":
    main()
