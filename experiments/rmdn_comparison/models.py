"""
This file holds models for comparison with RMDN.
"""

from keras.models import Model
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten
from keras.utils.data_utils import get_file


C3D_WEIGHTS_URL = 'http://imagelab.ing.unimore.it/files/c3d_weights/c3d-sports1M_weights.h5'


def C3DEncoder(input_shape, pretrained=True, summary=True):
    """
    Builds and returns an encoder based on the C3D network.

    :param input_shape: should be (3, 16, h, w).
    :param pretrained: optional, whether to load weights or not.
    :param summary: optional, whether to print summary or not.
    :return: a Keras model.
    """
    
    # define network input
    c3d_in = Input(input_shape)
    
    # 1st layer group
    h = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1')(c3d_in)
    h = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(h)

    # 2nd layer group
    h = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2')(h)
    h = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(h)

    # 3rd layer group
    h = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a')(h)
    h = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b')(h)
    h = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(h)

    # 4th layer group
    h = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a')(h)
    h = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b')(h)
    h = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4')(h)

    # 5th layer group
    h = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5a')(h)
    h = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5b')(h)

    c3d_encoding = Flatten()(h)
    model = Model(input=c3d_in, output=c3d_encoding, name='C3DEncoder')

    # load weights
    if pretrained:
        weights_path = get_file('c3d_encoder.h5', C3D_WEIGHTS_URL, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    # print summary
    if summary:
        model.summary()

    return model


# helper entry point to test models
if __name__ == '__main__':

    model = C3DEncoder(input_shape=(3, 16, 112, 112))
