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
from utils import get_torch_vars, compute_loss, evaluation_metrics # run_validation_step
from loss import CrossEntropyLabelSmooth
import time
import numpy as np
import cv2
from models import simple, Zhang_model
from dataset import *
import pickle

def main():
    # init the args
    args = Options().parse()
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    args.plot = False
    args.dataset = 'data'
    args.model = 'UNet'
    args.batch_size = 8  # 128 #32
    args.loss = 'CrossEntropyLoss'
    args.backbone = 'resnet18'

    args.lr = 1e-4  # 0.004  #0.01 #
    args.epochs = 10  # 60#20 #60
    args.lr_step = 15

    args.basepath = './'

    args.classification = True
    args.skip_connection = False
    args.eval = False
    args.from_npy = True

    # plot
    if args.plot:
        print('=>Enabling matplotlib for display:')
        plot.ion()
        plot.show()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Load data from saved .npy files...')
    print('Note for landscape data, these are Lab type.')
    x_train = np.load('./data/landscape/x_train.npy')
    y_train = np.load('./data/landscape/y_train.npy')
    y_train = y_train + 128

    x_valid = np.load('./data/landscape/x_valid.npy')
    # y_valid = np.load('./data/landscape/y_valid.npy')
    test_RGB_gt = np.load('./data/landscape/rgb_valid.npy')

    # init the model
    model = simple(args)
    if args.cuda:
        model.cuda()

    if args.classification:
        criterion = nn.CrossEntropyLoss()
        train_a = np.floor((y_train) / 13)[:,0,:,:]
        train_b = np.floor((y_train) / 13)[:,1,:,:]
        model_path = './checkpoints/landscape/classification_best_model.pth'
    else:
        criterion = nn.MSELoss()
        model_path = './checkpoints/landscape//regression_best_model.pth'

    def train(model):
        print("Beginning training ...")
        train_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_loss = float('inf')
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for i in range(0, x_train.shape[0], args.batch_size):
                # Regression Training
                if not args.classification:
                    batch_grey = torch.autograd.Variable(torch.from_numpy(x_train[i:i+args.batch_size]).float(), requires_grad = False)
                    batch_ab = torch.autograd.Variable(torch.from_numpy(y_train[i:i + args.batch_size]).float(),requires_grad=False)
                    if args.cuda:
                        batch_ab = batch_ab.cuda()
                        batch_grey = batch_grey.cuda()
                    optimizer.zero_grad()
                    batch_output = model(batch_grey)
                    loss = criterion(batch_output, batch_ab)
                # Classification Training
                else:
                    batch_grey = torch.autograd.Variable(torch.from_numpy(x_train[i:i+args.batch_size]).float(), requires_grad = False)
                    batch_a = torch.autograd.Variable(torch.from_numpy(train_a[i:i + args.batch_size]).long(), requires_grad=False)
                    batch_b = torch.autograd.Variable(torch.from_numpy(train_b[i:i + args.batch_size]).long(), requires_grad=False)
                    if args.cuda:
                        batch_grey = batch_grey.cuda()
                        batch_a = batch_a.cuda()
                        batch_b = batch_b.cuda()
                    optimizer.zero_grad()
                    batch_output = model(batch_grey)
                    loss_a = criterion(batch_output[0], batch_a)
                    loss_b = criterion(batch_output[1], batch_b)
                    loss = loss_a + loss_b
                loss.backward()
                optimizer.step()
                losses.append(loss.data.item())
            avg_loss = np.mean(losses)
            train_loss.append(avg_loss)
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, args.epochs, avg_loss))

            if avg_loss < best_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = avg_loss
                print('Best Training Model Saved')
        return model

    def test(model):
        model.eval()
        eval_loss_test = []
        for i in range(0, x_valid.shape[0], args.batch_size):
            test_L_input = torch.autograd.Variable(torch.from_numpy(x_valid[i:i+args.batch_size]).float(), requires_grad=False)
            if args.cuda:
                test_L_input = test_L_input.cuda()
            test_ab = model(test_L_input)
            if not args.classification:
                if args.cuda:
                    pred_rgb = cvt2RGB(test_L_input.cpu().data.numpy(), test_ab.cpu().data.numpy(), args.classification)
                else:
                    pred_rgb = cvt2RGB(test_L_input.data.numpy(), test_ab.data.numpy(), args.classification)
            else:
                if args.cuda:
                    test_ab = (test_ab[0].cpu().data.numpy(), test_ab[1].cpu().data.numpy())
                    pred_rgb = cvt2RGB(test_L_input.cpu().data.numpy(), test_ab, args.classification)
                else:
                    test_ab = (test_ab[0].data.numpy(), test_ab[1].data.numpy())
                    pred_rgb = cvt2RGB(test_L_input.data.numpy(), test_ab, args.classification)
            np.save('./pred_rgb_example.npy', pred_rgb)
            gt_rgb = test_RGB_gt[i:i+args.batch_size]
            np.save('./compare_example.npy', gt_rgb)
            eval_loss = evaluation_metrics(pred_rgb, gt_rgb)
            eval_loss_test.append(eval_loss)
        eval_loss = np.mean(eval_loss_test)
        print('Average Deviation Level in Color Saturation is %.4f ' % eval_loss)

    if not args.eval:
        model = train(model)
    else:
        model.load_state_dict(torch.load(model_path))
        test(model)

if __name__ == '__main__':
    main()