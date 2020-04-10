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

class simple(nn.Module):
    def __init__(self):
        super(simple, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Zhang_model(nn.Module):
    def __init__(self):
        super(Zhang_model, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(128)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(256)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(512)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(512)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(512)
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(512)
        )
        self.layer8up=torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.layer3to8=torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.layer8 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(256)
        )

        self.layer9up = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.layer2to9 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.layer9 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(128)
        )

        self.layer10up = torch.nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.layer1to10 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.layer10 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.classification_U = torch.nn.Conv2d(256, 50, kernel_size=1, padding=0, dilation=1, stride=1)
        self.classification_V = torch.nn.Conv2d(256, 50, kernel_size=1, padding=0, dilation=1, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
