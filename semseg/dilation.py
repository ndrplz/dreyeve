# -*- coding: utf-8 -*-
"""Dilation models for Keras.

# Reference:


- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/pdf/1511.07122.pdf)

"""

import warnings

from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Input, AtrousConvolution2D
from keras.layers import Dropout, UpSampling2D, ZeroPadding2D, Permute, Reshape, Activation
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
import numpy as np
import numba
import cv2


def softmax(x, restore_shape=True):
    """
    Softmax activation for a tensor x. No need to unroll the input first.
    :param x: x is a tensor with shape (None, channels, h, w)
    :param restore_shape: if False, output is returned unrolled (None, h * w, channels)
    :return: softmax activation of tensor x
    """
    _, c, h, w = x._keras_shape
    x = Permute(dims=(2, 3, 1))(x)
    x = Reshape(target_shape=(h * w, c))(x)

    x = Activation('softmax')(x)

    if restore_shape:
        x = Reshape(target_shape=(h, w, c))(x)
        x = Permute(dims=(3, 1, 2))(x)

    return x

# configuration for different datasets
CONFIG = {
    'cityscapes': {
        'classes': 19,
        'weights_file': 'dilation_cityscapes.h5',
        'weights_url': 'http://imagelab.ing.unimore.it/files/dilation_keras/cityscapes.h5',
        'input_shape': (3, 1396, 1396),
        'test_image': 'imgs_dilation/cityscapes.png',
        'mean_pixel': (72.39, 82.91, 73.16),
        'palette': np.array([[128, 64, 128],
                            [244, 35, 232],
                            [70, 70, 70],
                            [102, 102, 156],
                            [190, 153, 153],
                            [153, 153, 153],
                            [250, 170, 30],
                            [220, 220, 0],
                            [107, 142, 35],
                            [152, 251, 152],
                            [70, 130, 180],
                            [220, 20, 60],
                            [255, 0, 0],
                            [0, 0, 142],
                            [0, 0, 70],
                            [0, 60, 100],
                            [0, 80, 100],
                            [0, 0, 230],
                            [119, 11, 32]], dtype='uint8'),
        'zoom': 1,
        'conv_margin': 186
    },
    'voc12': {
        'classes': 21,
        'weights_file': 'dilation_voc12.h5',
        'weights_url': 'http://imagelab.ing.unimore.it/files/dilation_keras/voc12.h5',
        'input_shape': (3, 900, 900),
        'test_image': 'imgs_dilation/voc.jpg',
        'mean_pixel': (102.93, 111.36, 116.52),
        'palette': np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128]], dtype='uint8'),
        'zoom': 8,
        'conv_margin': 186
    },
    'kitti': {
        'classes': 11,
        'weights_file': 'dilation_kitti.h5',
        'weights_url': 'http://imagelab.ing.unimore.it/files/dilation_keras/kitti.h5',
        'input_shape': (3, 852, 1640),
        'test_image': 'imgs_dilation/kitti.png',
        'mean_pixel': (96.19, 95.55, 91.34),
        'palette': np.array([[128, 0, 0],
                             [128, 128, 0],
                             [128, 128, 128],
                             [64, 0, 128],
                             [192, 128, 128],
                             [128, 64, 128],
                             [64, 64, 0],
                             [64, 64, 128],
                             [192, 192, 128],
                             [0, 0, 192],
                             [0, 128, 192]], dtype='uint8'),
        'zoom': 8,
        'conv_margin': 186
    },
    'camvid': {
        'classes': 11,
        'weights_file': 'dilation_camvid.h5',
        'weights_url': 'http://imagelab.ing.unimore.it/files/dilation_keras/camvid.h5',
        'input_shape': (3, 900, 1100),
        'test_image': 'imgs_dilation/camvid.png',
        'mean_pixel': (110.70, 108.77, 105.41),
        'palette': np.array([[128, 0, 0],
                             [128, 128, 0],
                             [128, 128, 128],
                             [64, 0, 128],
                             [192, 128, 128],
                             [128, 64, 128],
                             [64, 64, 0],
                             [64, 64, 128],
                             [192, 192, 128],
                             [0, 0, 192],
                             [0, 128, 192]], dtype='uint8'),
        'zoom': 8,
        'conv_margin': 186
    }
}


# this function is the same as the one in the original repository
# basically it performs upsampling for datasets having zoom > 1
@numba.jit(nopython=True)
def interp_map(prob, zoom, width, height):
    zoom_prob = np.zeros((prob.shape[0], height, width), dtype=np.float32)
    for c in range(prob.shape[0]):
        for h in range(height):
            for w in range(width):
                r0 = h // zoom
                r1 = r0 + 1
                c0 = w // zoom
                c1 = c0 + 1
                rt = float(h) / zoom - r0
                ct = float(w) / zoom - c0
                v0 = rt * prob[c, r1, c0] + (1 - rt) * prob[c, r0, c0]
                v1 = rt * prob[c, r1, c1] + (1 - rt) * prob[c, r0, c1]
                zoom_prob[c, h, w] = (1 - ct) * v0 + ct * v1
    return zoom_prob


# CITYSCAPES MODEL
def get_dilation_model_cityscapes(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(32, 32))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)
    h = ZeroPadding2D(padding=(64, 64))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    h = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    # the following two layers pretend to be a Deconvolution with grouping layer.
    # never managed to implement it in Keras
    # since it's just a gaussian upsampling trainable=False is recommended
    h = UpSampling2D(size=(8, 8))(h)
    logits = Convolution2D(classes, 16, 16, border_mode='same', bias=False, trainable=False, name='ctx_upsample')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_cityscapes')

    return model


# PASCAL VOC MODEL
def get_dilation_model_voc(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, activation='relu', name='fc-final')(h)
    h = ZeroPadding2D(padding=(33, 33))(h)
    h = Convolution2D(2 * classes, 3, 3, activation='relu', name='ct_conv1_1')(h)
    h = Convolution2D(2 * classes, 3, 3, activation='relu', name='ct_conv1_2')(h)
    h = AtrousConvolution2D(4 * classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ct_conv2_1')(h)
    h = AtrousConvolution2D(8 * classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ct_conv3_1')(h)
    h = AtrousConvolution2D(16 * classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ct_conv4_1')(h)
    h = AtrousConvolution2D(32 * classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ct_conv5_1')(h)
    h = Convolution2D(32 * classes, 3, 3, activation='relu', name='ct_fc1')(h)
    logits = Convolution2D(classes, 1, 1, name='ct_final')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_voc12')

    return model


# KITTI MODEL
def get_dilation_model_kitti(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    logits = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_kitti')

    return model


# CAMVID MODEL
def get_dilation_model_camvid(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    logits = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_camvid')

    return model


# model function
def DilationNet(dataset, input_shape=None, apply_softmax=True, pretrained=True,
                input_tensor=None, classes=None):
    """ Instantiate the Dilation network architecture, optionally loading weights
        pre-trained on a dataset in the set (cityscapes, voc12, kitti, camvid).
        Note that pre-trained model is only available for Theano dim ordering.

        The model and the weights should be compatible with both
        TensorFlow and Theano backends.

        # Arguments
            dataset: choose among (cityscapes, voc12, kitti, camvid).
            input_shape: shape tuple. It should have exactly 3 inputs channels,
                and the axis ordering should be coherent with what specified in
                your keras.json (e.g. use (3, 512, 512) for 'th' and (512, 512, 3)
                for 'tf'). None will default to dataset specific sizes.
            apply_softmax: whether to apply softmax or return logits.
            pretrained: boolean. If `True`, loads weights coherently with
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of segmentation classes. If pretrained is True,
                it should be coherent with the dataset chosen.

        # Returns
            A Keras model instance.
    """
    if dataset not in {'cityscapes', 'voc12', 'kitti', 'camvid'}:
        raise ValueError('The `dataset` argument should be one among '
                         '(cityscapes, voc12, kitti, camvid)')

    if classes is not None:
        if classes != CONFIG[dataset]['classes'] and pretrained:
            raise ValueError('Cannot load pretrained model for dataset `{}` '
                             'with {} classes'.format(dataset, classes))
    else:
        classes = CONFIG[dataset]['classes']

    if input_shape is None:
        input_shape = CONFIG[dataset]['input_shape']

    # get the model
    if dataset == 'cityscapes':
        model = get_dilation_model_cityscapes(input_shape=input_shape, apply_softmax=apply_softmax,
                                              input_tensor=input_tensor, classes=classes)
    elif dataset == 'voc12':
        model = get_dilation_model_voc(input_shape=input_shape, apply_softmax=apply_softmax,
                                       input_tensor=input_tensor, classes=classes)
    elif dataset == 'kitti':
        model = get_dilation_model_kitti(input_shape=input_shape, apply_softmax=apply_softmax,
                                         input_tensor=input_tensor, classes=classes)
    elif dataset == 'camvid':
        model = get_dilation_model_camvid(input_shape=input_shape, apply_softmax=apply_softmax,
                                          input_tensor=input_tensor, classes=classes)

    # load weights
    if pretrained:
        if K.image_dim_ordering() == 'th':
            weights_path = get_file(CONFIG[dataset]['weights_file'],
                                    CONFIG[dataset]['weights_url'],
                                    cache_subdir='models')

            model.load_weights(weights_path)

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            raise NotImplementedError('Pretrained DilationNet model is not available with '
                                      'tensorflow dim ordering')

    return model


# predict function, mostly reported as it was in the original repo
def predict(image, model, ds):

    image = image.astype(np.float32) - 128
    conv_margin = CONFIG[ds]['conv_margin']

    input_dims = (1,) + CONFIG[ds]['input_shape']
    batch_size, num_channels, input_height, input_width = input_dims
    model_in = np.zeros(input_dims, dtype=np.float32)

    image_size = image.shape
    output_height = input_height - 2 * conv_margin
    output_width = input_width - 2 * conv_margin
    image = cv2.copyMakeBorder(image, conv_margin, conv_margin,
                               conv_margin, conv_margin,
                               cv2.BORDER_REFLECT_101)

    num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
    num_tiles_w = image_size[1] // output_width + (1 if image_size[1] % output_width else 0)

    model_in = np.zeros((num_tiles_h*num_tiles_w,)+CONFIG[ds]['input_shape'], dtype=np.float32)
    for h in range(num_tiles_h):
        for w in range(num_tiles_w):
            offset = [output_height * h,
                      output_width * w]
            tile = image[offset[0]:offset[0] + input_height,
                   offset[1]:offset[1] + input_width, :]
            margin = [0, input_height - tile.shape[0],
                      0, input_width - tile.shape[1]]
            tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                      margin[2], margin[3],
                                      cv2.BORDER_REFLECT_101)
            model_in[h*num_tiles_w+w] = tile.transpose([2, 0, 1])

    prob = model.predict(model_in)

    row_prediction = []
    for h in range(num_tiles_h):
        col_prediction = []
        for w in range(num_tiles_w):
            col_prediction.append(prob[h*num_tiles_w+w])
        col_prediction = np.concatenate(col_prediction, axis=2)
        row_prediction.append(col_prediction)
    prob = np.concatenate(row_prediction, axis=1)

    if CONFIG[ds]['zoom'] > 1:
        prob = interp_map(prob, CONFIG[ds]['zoom'], image_size[1], image_size[0])

    prediction = np.argmax(prob, axis=0)
    prediction = prediction[0:image_size[0], 0:image_size[1]]
    color_image = CONFIG[ds]['palette'][prediction.ravel()].reshape(image_size)

    return color_image


# predict function, no tiles
def predict_no_tiles(image, model, ds):

    image = image.astype(np.float32) - CONFIG[ds]['mean_pixel']
    conv_margin = CONFIG[ds]['conv_margin']

    input_dims = (1, 3, 1452, 2292)
    batch_size, num_channels, input_height, input_width = input_dims
    model_in = np.zeros(input_dims, dtype=np.float32)

    image_size = image.shape
    image = cv2.copyMakeBorder(image, conv_margin, conv_margin,
                               conv_margin, conv_margin,
                               cv2.BORDER_REFLECT_101)
    model_in[0] = image.transpose([2, 0, 1])

    prob = model.predict(model_in)

    return prob



if __name__ == '__main__':

    ds = 'camvid'  # choose between cityscapes, kitti, camvid, voc12

    # get the model
    model = DilationNet(dataset=ds)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()

    # read and predict a image
    im = cv2.imread(CONFIG[ds]['test_image'])
    y_img = predict(im, model, ds)

    # plot results
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(y_img)
    a.set_title('Semantic segmentation')
    plt.show(fig)
