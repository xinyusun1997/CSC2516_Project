import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Image Colorization')
        parser.add_argument('--dataset', type=str, default='data', help='training dataset (default: cifar10)')

        parser.add_argument('--loss', type=str,
                            default='CrossEntropyLoss',
                            help='Other Possible Loss Function: CrossEntropyLoss, FocalLoss, Magnetloss, OIMLoss')

        parser.add_argument('--n_codes', type=int, default=32,
                            metavar='N', help='(default: 32)')

        parser.add_argument('--block', type=int, default=3,
                            help='')

        parser.add_argument('--shuffle', action='store_true', default=False,
                            help='matplotlib')

        parser.add_argument('--nclass', type=int, default=24, metavar='N',
                            help='number of color classes (default: 24)')

        parser.add_argument('--backbone', type=str, default='resnet18',
                            help='backbone name (default: resnet18)')

        parser.add_argument('--batch-size', type=int, default=128,
                            metavar='N', help='batch size for training (default: 128)')

        parser.add_argument('--test-batch-size', type=int, default=1,
                            metavar='N', help='batch size for testing (default: 1)')

        parser.add_argument('--epochs', type=int, default=60, metavar='N',
                            help='number of epochs to train (default: 60)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')

        parser.add_argument('--plot', action='store_true', default=True,
                            help='matplotlib')

        parser.add_argument('--checkname', type=str, default='0', help='set the checkpoint name')


        # model params
        parser.add_argument('--model', type=str, default='pretrain',
            help='network model type (default: pretrain)')

        # lr setting
        parser.add_argument('--lr-scheduler', type=str, default='step',
            help='learning rate scheduler (default: step)')
        parser.add_argument('--lr-step', type=int, default=20, metavar='LR',
            help='learning rate step (default: 20)')

        # optimizer
        parser.add_argument('--momentum', type=float, default=0.9,
            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4,
            metavar ='M', help='SGD weight decay (default: 5e-4)')

        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true',
            default=False, help='disables CUDA training')

        parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
            help='put the path to resuming file if needed')

        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
            help='evaluating')

        parser.add_argument('--test_aug', action='store_true', default=False,
                            help='matplotlib')


        parser.add_argument('--ohem', type=int, default=-1,
                            help='')
        parser.add_argument('--atte', action='store_true', default=False,
                            help='')
        parser.add_argument('--mixup', action='store_true', default=False,
                            help='')
        parser.add_argument('--metric', type=str, default='',
                            help='arc_margin')
        parser.add_argument('--experiment_name', type=str, default='default_experiment',
                            help='experiment_name')
        parser.add_argument('--downsize_input', action='store_true', default= False,
            help='downsize_input')
        parser.add_argument('--colours', type=str, default='./data/colours/colour_kmeans24_cat7.npy',
                            help='colours categories path')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
