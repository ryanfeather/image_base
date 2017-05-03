import cv2
from keras.backend import backend
import subprocess
import os
import numpy as np
import pickle

import gzip

BACKEND = 'th' if backend()=='theano' else 'tf'

_data_folder = None

def set_data_base(base):
    global _data_folder
    _data_folder = base

def data_base(sub_path=None):
    if _data_folder is None:
        raise NotADirectoryError('data base is None, please set a valid directory with set_data_base')
    if sub_path is None:
        return _data_folder
    else:
        return os.path.join(_data_folder, sub_path)


def git_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()

def cv2k(image, single_dim_im=False):

    image_size = image.shape[-3:-1]
    if single_dim_im:
        new_shape = image.shape + (1,)
        result = image.reshape(new_shape)
    else:
        result = image

    long = len(image.shape) == 4

    if BACKEND == 'th':
        if long:
            result = image.transpose((0,3,1,2))
        else:
            result = image.transpose((2,0,1))
    else:
        result = result

    return result

def k2cv(image, single_dim_im=False):
    long = len(image.shape) == 4

    if BACKEND == 'th':
        if long:
            result = image.transpose((0,2,3,1))
        else:
            result = image.transpose((1,2,0))

    if single_dim_im:
        result = result.reshape(result.shape[:-1])

    return result


def get_im_cv2(path, resolution=None, ratio=None,size=None, upscale=True, downscale=True):
    """ OpenCV read image with a variety of scaling options

    :param path:
    :param resolution:
    :param ratio:
    :param size:
    :param upscale: increase resolution if needed
    :param downscale:  decrease resolution if needed
    :return:
    """
    img = cv2.imread(path)
    if resolution is not None or size is not None:
        if resolution is not None:
            if ratio is None:
                ratio = img.shape[1]/resolution[0]
            width = int(resolution*ratio)
            size = (width, resolution)

        do_resize = ((upscale and (size[0] > img.shape[1] or size[1] > img.shape[0])) or
                     (downscale and (size[0] < img.shape[1] or size[1] < img.shape[0])))

        if do_resize:
            resized = cv2.resize(img, size, cv2.INTER_LINEAR)
            return resized
    return img


def load_file(fl, resolution=None, size=None, upscale=True, downscale=True):
    flbase = os.path.basename(fl)
    img = get_im_cv2(fl, resolution=resolution, size=size, upscale=upscale, downscale=downscale).astype(np.float32)
    return img, flbase


def save_models(models, info_string):

    out_dir = os.path.join(data_base(), 'saved_models')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, info_string+'_mod_.pkl')
    with open(out_file, 'wb') as dump_file:
        pickle.dump(models, dump_file)

def load_models(info_string):
    out_dir = os.path.join(data_base(), 'saved_models')
    out_file = os.path.join(out_dir, info_string+'_mod_.pkl')
    with open(out_file, 'rb') as dump_file:
        models = pickle.load(dump_file)
    return models


def save_data(data, info_string, compress=True):
    out_dir = os.path.join(data_base(), 'saved_data')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, info_string + '_dat')
    if compress:
        fp =  gzip.GzipFile(out_file+'.npy.gz', 'w', compresslevel=1)
        np.save(fp, data)
        fp.close()
    else:
        np.save(out_file, data)


def load_data_from_path(path, compressed=True):

    if compressed:
        fp = gzip.GzipFile(path, 'r')
        result =  np.load(fp)
        return result
    else:
        return np.load(path)

def load_data(info_string, compressed=True):
    out_dir = os.path.join(data_base(), 'saved_data')
    out_file = os.path.join(out_dir, info_string+'_dat.npy')
    if compressed:
        fp = gzip.GzipFile(out_file+'.gz', 'r')
        result =  np.load(fp)
        return result
    else:
        return np.load(out_file)
