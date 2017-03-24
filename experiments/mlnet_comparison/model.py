from __future__ import division
from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
import keras.backend as K
import h5py
from eltwise_product import EltWiseProduct
import math
from config import shape_r_gt, shape_c_gt


def get_weights_vgg16(f, id):
    g = f['layer_{}'.format(id)]
    return [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]


def ml_net_model(img_rows=480, img_cols=640, downsampling_factor_net=8, downsampling_factor_product=10):
    f = h5py.File("vgg16_weights.h5")

    input_ml_net = Input(shape=(3, img_rows, img_cols))

    #########################################################
    # FEATURE EXTRACTION NETWORK							#
    #########################################################
    weights = get_weights_vgg16(f, 1)
    conv1_1 = Convolution2D(64, 3, 3, weights=weights, activation='relu', border_mode='same')(input_ml_net)
    weights = get_weights_vgg16(f, 3)
    conv1_2 = Convolution2D(64, 3, 3, weights=weights, activation='relu', border_mode='same')(conv1_1)
    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv1_2)

    weights = get_weights_vgg16(f, 6)
    conv2_1 = Convolution2D(128, 3, 3, weights=weights, activation='relu', border_mode='same')(conv1_pool)
    weights = get_weights_vgg16(f, 8)
    conv2_2 = Convolution2D(128, 3, 3, weights=weights, activation='relu', border_mode='same')(conv2_1)
    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv2_2)

    weights = get_weights_vgg16(f, 11)
    conv3_1 = Convolution2D(256, 3, 3, weights=weights, activation='relu', border_mode='same')(conv2_pool)
    weights = get_weights_vgg16(f, 13)
    conv3_2 = Convolution2D(256, 3, 3, weights=weights, activation='relu', border_mode='same')(conv3_1)
    weights = get_weights_vgg16(f, 15)
    conv3_3 = Convolution2D(256, 3, 3, weights=weights, activation='relu', border_mode='same')(conv3_2)
    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv3_3)

    weights = get_weights_vgg16(f, 18)
    conv4_1 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv3_pool)
    weights = get_weights_vgg16(f, 20)
    conv4_2 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv4_1)
    weights = get_weights_vgg16(f, 22)
    conv4_3 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv4_2)
    conv4_pool = MaxPooling2D((2, 2), strides=(1, 1), border_mode='same')(conv4_3)

    weights = get_weights_vgg16(f, 25)
    conv5_1 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv4_pool)
    weights = get_weights_vgg16(f, 27)
    conv5_2 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv5_1)
    weights = get_weights_vgg16(f, 29)
    conv5_3 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv5_2)


    #########################################################
    # ENCODING NETWORK										#
    #########################################################
    concatenated = merge([conv3_pool, conv4_pool, conv5_3], mode='concat', concat_axis=1)
    dropout = Dropout(0.5)(concatenated)

    int_conv = Convolution2D(64, 3, 3, init='glorot_normal', activation='relu', border_mode='same')(dropout)

    pre_final_conv = Convolution2D(1, 1, 1, init='glorot_normal', activation='relu')(int_conv)


    #########################################################
    # PRIOR LEARNING										#
    #########################################################
    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zero', W_regularizer=l2(1/(rows_elt*cols_elt)))(pre_final_conv)
    output_ml_net = Activation('relu')(eltprod)

    model = Model(input=[input_ml_net], output=[output_ml_net])

    return model


def loss(y_true, y_pred):
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r_gt, axis=-1)), shape_c_gt, axis=-1)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
