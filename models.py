import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MyConv2d(nn.Module):
    """
    Our simplified implemented of nn.Conv2d module for 2D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)


class UNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(UNet, self).__init__()

        ############### YOUR CODE GOES HERE ###############
        padding = kernel // 2
        self.block1 = nn.Sequential(
            MyConv2d(num_in_channels, num_filters, kernel, padding),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            MyConv2d(num_filters, 2*num_filters, kernel, padding),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=2*num_filters),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            MyConv2d(2*num_filters, 2*num_filters, kernel, padding),
            nn.BatchNorm2d(num_features=2*num_filters),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            MyConv2d(4*num_filters, num_filters, kernel, padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            MyConv2d(2*num_filters, num_colours, kernel, padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )

        self.final_layer = MyConv2d(num_in_channels+num_colours, num_colours, kernel, padding)
        ###################################################

    def forward(self, x):
        ############### YOUR CODE GOES HERE ###############
        self.hidden_1 = self.block1(x)
        self.hidden_2 = self.block2(self.hidden_1)
        self.hidden_3 = self.block3(self.hidden_2)
        self.hidden_4 = self.block4(torch.cat((self.hidden_2, self.hidden_3), 1))
        self.hidden_5 = self.block5(torch.cat((self.hidden_1, self.hidden_4), 1))
        self.output = self.final_layer(torch.cat((x, self.hidden_5), 1))
        return self.output
        ###################################################