# Data load functions adapted from CSC2516 A2 code
######################################################################
# Helper functions for loading data
######################################################################
# adapted from
# https://github.com/fchollet/keras/blob/master/keras/datasets/cifar10.py
import os
from keras.utils.data_utils import _extract_archive
from six.moves.urllib.request import urlretrieve
import tarfile
import numpy as np
import numpy.random as npr
import pickle
import sys
from PIL import Image
from skimage import io, color
from skimage.transform import rescale, resize

def process(xs, ys, max_pixel=256.0):
    """
    Pre-process CIFAR10 images
    Change inmages into HSV space

    Args:
      xs: the colour RGB pixel values
      ys: the category labels
      max_pixel: maximum pixel value in the original data
    Returns:
      xs: value normalized and shuffled colour images
      grey: greyscale images, also normalized so values are between 0 and 1
    """
    xs = xs / max_pixel
    xs = xs[np.where(ys == 7)[0], :, :, :]
    npr.shuffle(xs)

    grey = np.mean(xs, axis=1, keepdims=True)

    return (xs, grey)

def cvt2lab(rgb_data, classification=False, num_class=10):
    # rgb_data = rgb_data[np.where(ys == 7)[0], :, :, :]
    # Because skimage.color.rgb2lab expects the input values to be in [0, 1]
    rgb_data = rgb_data / 255
    npr.shuffle(rgb_data)
    # Change the order to make it suitable for LUV conversion
    rgb_data = np.rollaxis(rgb_data, 1, 4)
    LUV = np.empty(rgb_data.shape)
    if classification:
        a_onehot = np.empty((*rgb_data.shape[0:3], num_class))
        b_onehot = np.empty((*rgb_data.shape[0:3], num_class))
    for i in range(LUV.shape[0]):
        temp_rgb = rgb_data[i]
        temp_lab = color.rgb2lab(temp_rgb)
        temp_lab[:,:,1:] += 128
        if classification:
            temp_lab[:, :, 1:] = np.floor(temp_lab[:,:,1:] / np.ceil(256/num_class))
            for m in range(temp_lab.shape[0]):
                for n in range(temp_lab.shape[1]):
                    a_onehot[i, m, n] = np.eye(num_class)[np.int(temp_lab[m, n, 1])]
                    b_onehot[i, m, n] = np.eye(num_class)[np.int(temp_lab[m, n, 2])]
        LUV[i] = temp_lab

    if classification:
        return np.expand_dims(LUV[:,:,:,0], axis=1), (np.rollaxis(a_onehot, 3, 1), np.rollaxis(b_onehot, 3, 1))
    else:
        return np.expand_dims(LUV[:,:,:,0], axis=1), np.rollaxis(LUV[:,:,:,1:], 3, 1)

def cvt2lab_new(rgb_data, classification=False, num_class=10):
    # rgb_data = rgb_data[np.where(ys == 7)[0], :, :, :]
    # Because skimage.color.rgb2lab expects the input values to be in [0, 1]
    rgb_data = rgb_data / 255
    npr.shuffle(rgb_data)
    # Change the order to make it suitable for LUV conversion
    rgb_data = np.rollaxis(rgb_data, 1, 4)
    LUV = np.empty(rgb_data.shape)
    if classification:
        a_onehot = np.empty(rgb_data.shape[0:3])
        b_onehot = np.empty(rgb_data.shape[0:3])
    for i in range(LUV.shape[0]):
        temp_rgb = rgb_data[i]
        temp_lab = color.rgb2lab(temp_rgb)
        temp_lab[:,:,1:] += 128
        if classification:
            temp_lab[:, :, 1:] = np.floor(temp_lab[:,:,1:] / np.ceil(256/num_class))
            for m in range(temp_lab.shape[0]):
                for n in range(temp_lab.shape[1]):
                    a_onehot[i, m, n] = np.int(temp_lab[m, n, 1])
                    b_onehot[i, m, n] = np.int(temp_lab[m, n, 2])
        LUV[i] = temp_lab

    if classification:
        return np.expand_dims(LUV[:,:,:,0], axis=1), (a_onehot, b_onehot)
    else:
        return np.expand_dims(LUV[:,:,:,0], axis=1), np.rollaxis(LUV[:,:,:,1:], 3, 1)

def cvt2RGB(L, ab, classification = False, num_class = 10):
    if classification:
        L = np.rollaxis(L, 1, 4)
        pred_a = np.expand_dims(np.argmax(np.rollaxis(ab[0], 1, 4), axis=3), axis = 3)
        pred_b = np.expand_dims(np.argmax(np.rollaxis(ab[1], 1, 4), axis=3), axis = 3)
        pred_a = pred_a * 26 + 13 - 128
        pred_b = pred_b * 26 + 13 - 128
        Lab = np.concatenate((L, pred_a, pred_b), axis = 3)
    else:
        Lab = np.concatenate((L, ab), axis=1)
        Lab = np.rollaxis(Lab, 1, 4)
        Lab[:, :, :, 1:] -= 128
    RGB = np.empty(Lab.shape)
    for i in range(Lab.shape[0]):
        temp_Lab = Lab[i]
        temp_RGB = color.lab2rgb(temp_Lab)
        RGB[i] = temp_RGB
    return RGB


def get_file(fname, origin, untar=False, extract=False, archive_format='auto', cache_dir='data'):
    datadir = os.path.join(cache_dir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    print('File path: %s' % fpath)
    if not os.path.exists(fpath):
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            print('Extracting file.')
            with tarfile.open(fpath) as archive:
                archive.extractall(datadir)
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_cifar10(transpose=False):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if transpose:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)


# Dataset Retrieved from https://github.com/cetinsamet/image-colorization
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def read_image(filename, size=(256, 256), training=False):
    img = io.imread(filename)
    real_size = img.shape
    if img.shape!=size and not training:
        img = resize(img, size, anti_aliasing=False)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], 2)
    return img, real_size[:2]


def cvt2Lab_ls(image):
    Lab = color.rgb2lab(image)
    return Lab[:, :, 0], Lab[:, :, 1:]  # L, ab

def cvt2rgb_ls(image):
    return color.lab2rgb(image)

def load_landscape_dataset():

    # Try to train and test model on 256 x 256 images.
    TRAIN_IMAGENAME_PATH = "./data/landscape/train.txt"
    VALID_IMAGENAME_PATH = "./data/landscape/valid.txt"

    GRAY_IMAGE_PATH = "./data/landscape/gray/"
    COLOR_256_IMAGE_PATH = "./data/landscape/color_256/"
    with open(TRAIN_IMAGENAME_PATH, 'r') as infile:
        train_imagename = [line.strip() for line in infile]
    with open(VALID_IMAGENAME_PATH, 'r') as infile:
        valid_imagename = [line.strip() for line in infile]

    # LOAD GRAY IMAGES
    gray_images = list()
    for gray_imagename in os.listdir(GRAY_IMAGE_PATH):
        gray_image, s   = read_image(GRAY_IMAGE_PATH+gray_imagename)
        gray_image      = (gray_imagename, cvt2Lab_ls(gray_image)[0])
        gray_images.append(gray_image)
    gray_images = sorted(gray_images, key=lambda x: x[0])
    print("-> gray images are loded")

    # SPLIT GRAY IMAGES TO TRAIN AND VALIDATION SETS
    x_train, x_valid = np.empty([1, 1, 256, 256]), np.empty([1, 1, 256, 256])
    for gray_imagename, gray_image in gray_images:
        if gray_imagename in train_imagename:
            x_train = np.concatenate([x_train, np.reshape(gray_image, (1,) + x_train.shape[1:])])
        if gray_imagename in valid_imagename:
            x_valid = np.concatenate([x_valid, np.reshape(gray_image, (1,) + x_valid.shape[1:])])
    x_train, x_valid = x_train[1:], x_valid[1:]
    print("-> gray images are splitted to datasets")
    print()

    # LOAD 64X64 COLOR IMAGES
    color64_images = list()
    for color64_imagename in os.listdir(COLOR_256_IMAGE_PATH):
        color64_image, s    = read_image(COLOR_256_IMAGE_PATH+color64_imagename, training=True)
        color64_image       = (color64_imagename, cvt2Lab_ls(color64_image)[1])
        color64_images.append(color64_image)
    color64_images  = sorted(color64_images, key=lambda x:x[0])
    print("-> 256x256 color images are loded")

    # SPLIT 64x64 COLOR IMAGES TO TRAIN AND VALIDATION SETS
    y_train, y_valid    = np.empty([1, 256, 256, 2]), np.empty([1, 256, 256, 2])
    for color64_imagename, color64_image in color64_images:
        if color64_imagename in train_imagename:
            y_train = np.concatenate([y_train, np.expand_dims(color64_image, axis=0)])
        if color64_imagename in valid_imagename:
            y_valid = np.concatenate([y_valid, np.expand_dims(color64_image, axis=0)])
    y_train, y_valid = np.rollaxis(y_train[1:], 3, 1), np.rollaxis(y_valid[1:], 3, 1)
    print("-> 256x256 color images are splitted to datasets")
    print()
    np.save('./data/landscape/x_train.npy', x_train)
    np.save('./data/landscape/y_train.npy', y_train)
    np.save('./data/landscape/x_valid.npy', x_valid)
    np.save('./data/landscape/y_valid.npy', y_valid)
if __name__ == '__main__':
    load_landscape_dataset()