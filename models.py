from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import warnings
import torch.nn.functional as F
import os
from torchvision import transforms
import sklearn.neighbors as sknn
from skimage.transform import resize
from skimage import color


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
        self.classification_out = torch.nn.Conv2d(256, 313, kernel_size=1, padding=0, dilation=1, stride=1)
        self.regression_out = torch.nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.classification:
            x = self.classification_out(x)
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


class NNEncLayer(object):
    ''' Layer which encodes ab map into Q colors
    OUTPUTS
        top[0].data     NxQ
    '''

    def __init__(self):
        self.NN = 32
        self.sigma = 0.5
        self.ENC_DIR = './resources/'
        self.nnenc = NNEncode(self.NN, self.sigma, km_filepath=os.path.join(self.ENC_DIR, 'pts_in_hull.npy'))

        self.X = 256
        self.Y = 256
        self.Q = self.nnenc.K

    def forward(self, x):
        #return np.argmax(self.nnenc.encode_points_mtx_nd(x), axis=1).astype(np.int32)
        encode=self.nnenc.encode_points_mtx_nd(x)
        max_encode=np.argmax(encode,axis=1).astype(np.int32)
        return encode,max_encode

    def reshape(self, bottom, top):
        top[0].reshape(self.N, self.Q, self.X, self.Y)


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''

    def __init__(self, NN, sigma, km_filepath='', cc=-1):
        if (check_value(cc, -1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = sknn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        P = pts_flt.shape[0]
        if (sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P, self.K))
            self.p_inds = np.arange(0, P, dtype='int')[:, na()]

        P = pts_flt.shape[0]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]
        self.pts_enc_flt[self.p_inds, inds] = wts

        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)
        return pts_enc_nd

    def decode_points_mtx_nd(self, pts_enc_nd, axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd, axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt, self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis=axis)
        return pts_dec_nd

    # def decode_1hot_mtx_nd(self, pts_enc_nd, axis=1, returnEncode=False):
    #     pts_1hot_nd = nd_argmax_1hot(pts_enc_nd, axis=axis)
    #     pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd, axis=axis)
    #     if (returnEncode):
    #         return (pts_dec_nd, pts_1hot_nd)
    #     else:
    #         return pts_dec_nd


# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if (np.array(inds).size == 1):
        if (inds == val):
            return True
    return False


def na():  # shorthand for new axis
    return np.newaxis


def flatten_nd_array(pts_nd, axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.dim()
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = tuple(np.concatenate((nax, np.array(axis).flatten()), axis=0).tolist())
    pts_flt = pts_nd.permute(axorder)
    pts_flt = pts_flt.contiguous().view(NPTS.item(), SHP[axis].item())
    return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.dim()
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices

    if (squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        NEW_SHP = SHP[nax].tolist()

        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = tuple(np.argsort(axorder).tolist())
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    return pts_out


def decode(data_l, conv8_313, rebalance=1):
    # print('data_l',type(data_l))
    # print('shape',data_l.shape)
    # np.save('data_l.npy',data_l)
    # data_l = data_l[0] + 50
    # data_l = data_l.cpu().data.numpy().transpose((1, 2, 0))
    data_l = np.rollaxis(data_l, 1, 4)
    # conv8_313 = conv8_313[0]
    enc_dir = './resources'
    conv8_313_rh = conv8_313 * rebalance
    # print('conv8',conv8_313_rh.size())
    # class8_313_rh = F.softmax(conv8_313_rh, dim=0).cpu().data.numpy().transpose((1, 2, 0))
    class8_313_rh = F.softmax(conv8_313_rh, dim=1)
    # np.save('class8_313.npy',class8_313_rh)
    # class8 = np.argmax(class8_313_rh, axis=-1)
    class8 = np.argmax(class8_313_rh, axis=1)
    # print('class8',class8.shape)
    cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
    # data_ab = np.dot(class8_313_rh, cc)
    # data_ab = cc[class8[:][:]]
    data_ab = cc[class8]
    # data_ab=np.transpose(data_ab,axes=(1,2,0))
    # data_l=np.transpose(data_l,axes=(1,2,0))
    # data_ab = resize(data_ab, (224, 224,2))
    # data_ab = data_ab.repeat(4, axis=0).repeat(4, axis=1)
    # print(data_l.shape)
    # print(data_ab.shape)
    img_lab = np.concatenate((data_l, data_ab), axis=-1)
    # print(img_lab.shape)
    img_rgb = np.empty(img_lab.shape)
    for i in range(img_rgb.shape[0]):
        img_rgb[i] = color.lab2rgb(img_lab[i])
    return img_rgb

if __name__ == '__main__':
    img_l = np.load('./data/landscape/x_valid.npy')
    img_ab = np.load('./data/landscape/y_valid.npy')
    img_l = img_l[0:4]
    img_ab = torch.Tensor(img_ab[0:4]).float()
    encode_layer = NNEncLayer()
    encode, max_encode = encode_layer.forward(img_ab)
    np.save('./encode.npy', encode)
    np.save('./max_encode.npy', max_encode)
    temp = decode(img_l, torch.Tensor(encode).float())
    np.save('./new_result.npy', temp)