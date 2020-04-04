##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helper functions adopted from Hang Zhang's implementation

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

__all__ = ['get_optimizer', 'LR_Scheduler', 'save_checkpoint', 'progress_bar', 'CrossEntropyLabelSmooth', 'FocalLoss',
           'CenterLoss_5', 'ArcMarginProduct']


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.001, use_gpu=True, ignore_index=-1, reduce=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """

        index_pos = (targets != self.ignore_index).nonzero().squeeze()
        # Ohem Problem: Last batch contains less items, Error
        # if index_pos >= inputs.shape[0]:
        #     index_pos = 0
        # --------------------------------------------------
        targets = targets[index_pos]
        inputs = inputs[index_pos, :]

        try:
            # print(inputs)
            log_probs = self.logsoftmax(inputs)
        # print(inputs)
        except RuntimeError:
            print(index_pos)
            print(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.reduce:
            loss = (- targets * log_probs).mean(0).sum()  # * torch.from_numpy(np.array([1.0 , 4.0])).cuda().float()
        else:
            loss = (- targets * log_probs).mean(1)  # * torch.from_numpy(np.array([1.0 , 4.0])).cuda().float()
            # print(loss.size())
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True, num_classes=None, epsilon=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # if isinstance(alpha, Variable):
        self.alpha = Variable(alpha * torch.ones(num_classes, 1))
        # else:
        #     self.alpha = Variable(alpha)
        self.size_average = size_average
        self.class_num = num_classes

    def forward(self, inputs, target):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class CenterLoss_5(nn.Module):
    """

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):  # , margin = 0.3):
        super(CenterLoss_5, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        # self.margin = margin

    def forward(self, x, labels, center_weight):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        labels_copy = labels.clone()
        self.centers = center_weight
        # print(center_weight)
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        distmat = distmat / 2048
        # distmat = distmat * 0.002

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        # center inter method:2
        uni_labels = torch.unique(labels_copy.cpu()).cuda()
        n = len(uni_labels)

        self.centers_norm = 1. * self.centers / (
                torch.norm(self.centers, 2, dim=-1, keepdim=True).expand_as(self.centers) + 1e-12)

        # print( self.centers[uni_labels, :] * (self.centers[uni_labels, :].t()))
        loss_center_inter = torch.abs(self.centers_norm[uni_labels, :].mm(self.centers_norm[uni_labels, :].t()) \
                                      * (1 - torch.eye(n).cuda())).mean()
        # print(loss_center_inter)
        #
        loss_all = loss + 1.0 * loss_center_inter

        return loss_all


# ArcFace
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # Parameter 的用途：
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        # net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的
        # https://www.jianshu.com/p/d8b77cc02410
        # 初始化权重
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将cos(\theta + m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


def get_optimizer(args, model, diff_LR=True):
    """
    Returns an optimizer for given model,

    Args:
        args: :attr:`args.lr`, :attr:`args.momentum`, :attr:`args.weight_decay`
        model: if using different lr, define `model.pretrained` and `model.head`.
    """
    if diff_LR and model.pretrained is not None:
        print('Using different learning rate for pre-trained features')
        optimizer = torch.optim.SGD([
            {'params': model.pretrained.parameters()},
            {'params': model.head.parameters(),
             'lr': args.lr * 10},
        ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    else:
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    return optimizer


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`), :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs, :attr:`args.lr_step`

        niters: number of iterations per epoch
    """

    def __init__(self, args, niters=420):
        self.mode = args.lr_scheduler
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = args.lr
        if self.mode == 'step':
            self.lr_step = args.lr_step
        else:
            self.niters = niters
            self.N = args.epochs * niters
        self.epoch = -1

    def __call__(self, optimizer, i, epoch, best_pred):
        if self.mode == 'cos':
            T = (epoch - 1) * self.niters + i
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
            # print(lr)
        elif self.mode == 'poly':
            T = (epoch - 1) * self.niters + i
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** ((epoch - 1) // self.lr_step))
        else:
            raise RuntimeError('Unknown LR scheduler!')
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (
                epoch, lr, best_pred))
            self.epoch = epoch
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, args, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "%s/runs/%s/%s/%s/" % (args.basepath, args.dataset, args.model, args.checkname)
    # print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


# refer to https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
# terms_height, term_width = os.popen('stty size', 'r').read().split()
term_width = 2
term_width = int(term_width) - 1
TOTAL_BAR_LENGTH = 36.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    """Progress Bar for display
    """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('    Step: %s' % _format_time(step_time))
    L.append(' | Tot: %s' % _format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def _format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


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

def run_validation_step(cnn, criterion, test_grey, test_rgb_cat, batch_size,
                        colours, plotpath=None, visualize=True, downsize_input=False):
    correct = 0.0
    total = 0.0
    losses = []
    num_colours = np.shape(colours)[0]
    for i, (xs, ys) in enumerate(get_batch(test_grey,
                                           test_rgb_cat,
                                           batch_size)):
        images, labels = get_torch_vars(xs, ys, args.gpu)
        outputs = cnn(images)

        val_loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colours=num_colours)
        losses.append(val_loss.data.item())

        _, predicted = torch.max(outputs.data, 1, keepdim=True)
        total += labels.size(0) * 32 * 32
        correct += (predicted == labels.data).sum()

    if plotpath: # only plot if a path is provided
        plot(xs, ys, predicted.cpu().numpy(), colours,
             plotpath, visualize=visualize, compare_bilinear=downsize_input)

    val_loss = np.mean(losses)
    val_acc = 100 * correct / total
    return val_loss, val_acc
