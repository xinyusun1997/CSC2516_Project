import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class simple(nn.Module):
    def __init__(self, args):
        super(simple, self).__init__()
        self.classification = args.classification
        self.skip_connection = args.skip_connection
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
            torch.nn.BatchNorm2d(256)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.classification_a = torch.nn.Conv2d(256, 10, kernel_size=1, padding=0, dilation=1, stride=1)
        self.classification_b = torch.nn.Conv2d(256, 10, kernel_size=1, padding=0, dilation=1, stride=1)

        self.regression_out = torch.nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.classification:
            x = (self.classification_a(x), self.classification_b(x))
        else:
            x = self.layer4(x)
            x = self.regression_out(x)
        return x

class Zhang_model(nn.Module):
    def __init__(self, args):
        super(Zhang_model, self).__init__()
        self.classification = args.classification
        self.skip_connection = args.skip_connection
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
        self.classification_a = torch.nn.Conv2d(256, 10, kernel_size=1, padding=0, dilation=1, stride=1)
        self.classification_b = torch.nn.Conv2d(256, 10, kernel_size=1, padding=0, dilation=1, stride=1)

        self.regression_out = torch.nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8up(x)
        x = self.layer8(x)
        x = self.layer9up(x)
        x = self.layer9(x)
        x = self.layer10up(x)
        x = self.layer10(x)
        x = self.regression_out(x)
        return x
