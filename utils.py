import shutil
import os
import sys
import time
import math
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from six.moves.urllib.request import urlretrieve
import matplotlib.pyplot as plot
from dataset import *
from skimage.transform import rescale, resize
from skimage import color


def get_torch_vars(xs, ys, gpu=False):
    """
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tenosor): greyscale input
      ys (int numpy tenosor): categorical labels
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      Variable(xs), Variable(ys)
    """
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).long()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return Variable(xs), Variable(ys)

def compute_loss(criterion, outputs, labels, batch_size, num_colours):
    """
    Helper function to compute the loss. Since this is a pixelwise
    prediction task we need to reshape the output and ground truth
    tensors into a 2D tensor before passing it in to the loss criteron.

    Args:
      criterion: pytorch loss criterion
      outputs (pytorch tensor): predicted labels from the model
      labels (pytorch tensor): ground truth labels
      batch_size (int): batch size used for training
      num_colours (int): number of colour categories
    Returns:
      pytorch tensor for loss
    """

    loss_out = outputs.transpose(1,3) \
                      .contiguous() \
                      .view([batch_size*32*32, num_colours])
    loss_lab = labels.transpose(1,3) \
                      .contiguous() \
                      .view([batch_size*32*32])
    return criterion(loss_out, loss_lab)

def evaluation_metrics(output_RGB, gt_RGB):
    eval_loss = []
    for i in range(0, gt_RGB.shape[0]):
        gt_hsv = color.rgb2hsv(gt_RGB[i])
        pred_hsv = color.rgb2hsv(output_RGB[i])
        if np.sum(gt_hsv[:, :, 1]) != 0:
            diff = abs(np.sum(gt_hsv[:, :, 1]) - np.sum(pred_hsv[:, :, 1])) / np.sum(gt_hsv[:, :, 1])
            eval_loss.append(diff)
    avg_diff = np.mean(eval_loss)
    return avg_diff

# def run_validation_step(cnn, criterion, test_grey, test_rgb_cat, batch_size,
#                         colours, plotpath=None, visualize=True, downsize_input=False):
#     correct = 0.0
#     total = 0.0
#     losses = []
#     num_colours = np.shape(colours)[0]
#     for i, (xs, ys) in enumerate(get_batch(test_grey,
#                                            test_rgb_cat,
#                                            batch_size)):
#         images, labels = get_torch_vars(xs, ys, args.gpu)
#         outputs = cnn(images)
#
#         val_loss = compute_loss(criterion,
#                                 outputs,
#                                 labels,
#                                 batch_size=args.batch_size,
#                                 num_colours=num_colours)
#         losses.append(val_loss.data.item())
#
#         _, predicted = torch.max(outputs.data, 1, keepdim=True)
#         total += labels.size(0) * 32 * 32
#         correct += (predicted == labels.data).sum()
#
#     if plotpath: # only plot if a path is provided
#         plot(xs, ys, predicted.cpu().numpy(), colours,
#              plotpath, visualize=visualize, compare_bilinear=downsize_input)
#
#     val_loss = np.mean(losses)
#     val_acc = 100 * correct / total
#     return val_loss, val_acc
#
