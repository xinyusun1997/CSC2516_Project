import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as resnet

# Below is an example model which I previously used for image classification task.
# We will build our own model based on this.
# Other exmaple model can be directly imported from torchvision.models

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

        self.linear = nn.Linear(512, 512)
        self.head = nn.Sequential(
            # nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1),
            # nn.Dropout(p=0.5)
        )

        self.head.apply(weights_init_kaiming)

        self.classifier = nn.Linear(512,nclass)
        self.classifier.apply(weights_init_classifier)

        self.softmax = nn.Softmax(1).cuda()

        if self.atte:
            self.c_atte = nn.Sequential(
                                        nn.Linear(128, 32, bias = True),
                                        # nn.BatchNorm1d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(32, 128, bias = True),
                                    )
            self.s_atte = nn.Sequential(
                                        nn.Conv2d(128,32,kernel_size=1,stride=1, bias = True),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True),
                                        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                        # nn.ReLU(inplace=True),
                                        nn.Conv2d(32,8,kernel_size=1,stride=1, bias = True),
                                        nn.BatchNorm2d(8),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(8,1,kernel_size=1,stride=1, bias = True),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True),
                                    )
            self.fuse_atte = nn.Sequential(
                                            nn.Conv2d(128,128,kernel_size=1,stride=1),
                                            nn.Sigmoid()
                                        )
            self.c_atte.apply(weights_init_kaiming)
            self.s_atte.apply(weights_init_kaiming)
            self.fuse_atte.apply(weights_init_kaiming)

            self.c_atte_0 = nn.Sequential(
                                        nn.Linear(64, 16, bias = True),
                                        # nn.BatchNorm1d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(16, 64, bias = True),
                                    )
            self.s_atte_0 = nn.Sequential(
                                        nn.Conv2d(64,16,kernel_size=1,stride=1, bias = True),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(inplace=True),
                                        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                        # nn.ReLU(inplace=True),
                                        nn.Conv2d(16,4,kernel_size=1,stride=1, bias = True),
                                        nn.BatchNorm2d(4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(4,1,kernel_size=1,stride=1, bias = True),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True),
                                    )
            self.fuse_atte_0 = nn.Sequential(
                                            nn.Conv2d(64,64,kernel_size=1,stride=1),
                                            nn.Sigmoid()
                                        )
            self.c_atte_0.apply(weights_init_kaiming)
            self.s_atte_0.apply(weights_init_kaiming)
            self.fuse_atte_0.apply(weights_init_kaiming)


        # self.dropblock = LinearScheduler(
        #     DropBlock2D(drop_prob=0.5, block_size=7),
        #     start_value=0.,
        #     stop_value=0.9,
        #     nr_steps=10)

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))

        # self.dropblock.step()
        # print(x)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        # print(x)
        # print(torch.sum(torch.abs(x -0.355846) < 0.00001))
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        # x = self.dropblock(x)
        if self.atte:
            C = F.max_pool2d(x, x.size()[2:])
            C = C.view(C.size(0), -1)
            C = self.c_atte_0(C).unsqueeze(2).unsqueeze(3) # N * c * 1 *1
            S = self.s_atte_0(x) # N * 1 * h *w
            A = S * C
            A = self.fuse_atte_0(A)
            x = A * x

        x = self.pretrained.layer2(x)

        if self.atte:
            C = F.max_pool2d(x, x.size()[2:])
            C = C.view(C.size(0), -1)
            C = self.c_atte(C).unsqueeze(2).unsqueeze(3) # N * c * 1 *1
            S = self.s_atte(x) # N * 1 * h *w
            A = S * C
            A = self.fuse_atte(A)
            x = A * x
        # x = self.dropblock(x)
        x = self.pretrained.layer3(x)
        # if self.atte:
        #     x = A * x
        x = self.pretrained.layer4(x)

        x1 = F.max_pool2d(x, x.size()[2:])
        x2 = F.avg_pool2d(x, x.size()[2:])
        x = x1 + x2

        x = x.view(x.size(0), -1)
        f = self.linear(x)
        x = self.head(f)

        # import numpy as np
        # store_img = x.detach().cpu().numpy()
        # np.save("data_pytorch.npy", store_img)
        x= self.classifier(x)
        return x, f, self.classifier.weight, self.softmax(x)
