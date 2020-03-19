import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as resnet

# Wait for future completion.

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Net(nn.Module):
    def __init__(self, args):
        nclass=args.nclass
        self.atte = args.atte
        super(Net, self).__init__()
        self.backbone = args.backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True)
        elif self.backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=True)
        elif self.backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=True)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))