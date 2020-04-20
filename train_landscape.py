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
from utils import get_torch_vars, compute_loss, evaluation_metrics, ssim_compute, psnr_compute # run_validation_step
import time
import numpy as np
import cv2
from models import simple, NNEncLayer, decode, simple_skip_connection
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
    args.batch_size = 4  # 128 #32
    args.loss = 'CrossEntropyLoss'
    args.backbone = 'resnet18'

    args.lr = 1e-4  # 0.004  #0.01 #
    args.epochs = 60  # 60#20 #60
    args.lr_step = 15

    args.basepath = './'

    args.classification = False
    args.skip_connection = True
    args.eval = True
    args.from_npy = True

    print('Load data from saved .npy files...')
    print('For landscape data, these are Lab type.')
    if not args.eval:
        x_train = np.load('./data/landscape/x_train.npy')
        y_train = np.load('./data/landscape/y_train.npy')
    x_valid = np.load('./data/landscape/x_valid.npy')
    test_RGB_gt = np.load('./data/landscape/rgb_valid.npy')
    print('Finish Data Loading')

    # init the model
    if args.skip_connection:
        model = simple_skip_connection(args)
    else:
        model = simple(args)
    if args.cuda:
        model.cuda()

    if args.classification:
        criterion = nn.CrossEntropyLoss()
        encode_layer = NNEncLayer()
        if args.skip_connection:
            model_path = './checkpoints/landscape/classification_with_sc.pth'
        else:
            model_path = './checkpoints/landscape/classification.pth'
    else:
        criterion = nn.MSELoss()
        if args.skip_connection:
            model_path = './checkpoints/landscape/regression_with_sc.pth'
        else:
            model_path = './checkpoints/landscape/regression.pth'


    def train(model):
        print("Beginning training ...")
        train_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_loss = -float('inf')
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
                    batch_ab = torch.autograd.Variable(torch.from_numpy(y_train[i:i + args.batch_size]).float(), requires_grad=False)
                    encode, max_encode = encode_layer.forward(batch_ab)
                    max_encode = torch.Tensor(max_encode).long()
                    if args.cuda:
                        batch_grey = batch_grey.cuda()
                        max_encode = max_encode.cuda()
                    optimizer.zero_grad()
                    batch_output = model(batch_grey)
                    loss = criterion(batch_output, max_encode)
                loss.backward()
                optimizer.step()
                losses.append(loss.data.item())

            (ssim_score, psnr) = test(model)

            avg_loss = np.mean(losses)
            train_loss.append(avg_loss)
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, args.epochs, avg_loss))

            if ssim_score > best_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = ssim_score
                print('Best Training Model Saved')

        if args.plot:
            plot.figure()
            plot.plot(train_loss, "ro-", label="Train")
            plot.legend()
            plot.title("CE Loss")
            plot.xlabel("Epochs")
            plot.savefig("./curve/training_curve.png")
        return model

    def test(model):
        print('Start Testing')
        model.eval()
        ssim_loss_test = []
        psnr_loss_test = []
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
                    # test_ab = decode(test_L_input.cpu().data.numpy(), test_ab.cpu().detach())
                    pred_rgb = decode(test_L_input.cpu().data.numpy(), test_ab.cpu().detach())
                else:
                    test_ab = (test_ab[0].data.numpy(), test_ab[1].data.numpy())
                    pred_rgb = cvt2RGB(test_L_input.data.numpy(), test_ab, args.classification)
            gt_rgb = test_RGB_gt[i:i+args.batch_size]
            # eval_loss = evaluation_metrics(pred_rgb, gt_rgb)
            ssim_loss = ssim_compute(pred_rgb, gt_rgb)
            ssim_loss_test.append(ssim_loss)
            psnr_loss = psnr_compute(pred_rgb, gt_rgb)
            psnr_loss_test.append(psnr_loss)
        ssim_loss = np.mean(ssim_loss_test)
        psnr_loss = np.mean(psnr_loss_test)
        print('Average SSIM score is %.4f ' % ssim_loss)
        print('Average PSNR is %.4f ' % psnr_loss)
        return (ssim_loss, psnr_loss)

    if not args.eval:
        train(model)
    else:
        model.load_state_dict(torch.load(model_path))
        test(model)

if __name__ == '__main__':
    main()