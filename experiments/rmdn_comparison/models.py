"""
This file holds models for comparison with RMDN.
"""

from keras.models import Model
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten, Dropout
from keras.layers import LSTM, Dense, TimeDistributedDense, Reshape, merge, Activation, Lambda
from keras.utils.data_utils import get_file
import keras.backend as K


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


def RMDN_train(hidden_states, n_mixtures, input_shape, summary=True):
    """
    Function that returns RMDN model for training. Here the recurrent layer has
    return_sequences=True and is stateless.

    :param hidden_states: dimension of the LSTM state.
    :param n_mixtures: number of output mixture densities.
    :param input_shape: the input shape like (time_steps, c3d_encoding).
    :param summary: optional, whether or not to print summary.
    :return: a Keras model.
    """
    time_steps, _ = input_shape

    sequence_in = Input(shape=input_shape, name='input')

    state = LSTM(output_dim=hidden_states,
                 return_sequences=True,
                 name='recurrent_module')(sequence_in)
    state = Dropout(0.5)(state)

    # Mixture Density Inference
    # mixture components weights
    weight = TimeDistributedDense(output_dim=n_mixtures * 1, name='output_weight')(state)
    weight = Reshape(target_shape=(time_steps, n_mixtures))(weight)
    weight = Activation('softmax')(weight)
    weight = Reshape(target_shape=(time_steps, n_mixtures, 1))(weight)

    # gaussian mean
    mu = TimeDistributedDense(output_dim=n_mixtures * 2, name='output_mean')(state)
    mu = Reshape(target_shape=(time_steps, n_mixtures, 2))(mu)
    mu = Activation('relu')(mu)  # this must become linear

    # variance
    sigma = TimeDistributedDense(output_dim=n_mixtures * 2, name='output_var')(state)
    sigma = Reshape(target_shape=(time_steps, n_mixtures, 2))(sigma)
    sigma = Lambda(lambda x: K.exp(x) + 1, output_shape=(time_steps, n_mixtures, 2))(sigma)

    # correlation
    ro = TimeDistributedDense(output_dim=n_mixtures, name='output_corr')(state)
    ro = Reshape(target_shape=(time_steps, n_mixtures, 1))(ro)
    ro = Activation('tanh')(ro)

    md = merge([weight, mu, sigma, ro], mode='concat', concat_axis=-1)

    model = Model(input=sequence_in, output=md)

    if summary:
        model.summary()

    return model


def RMDN_test(hidden_states, n_mixtures, input_shape, summary=True):
    """
    Function that returns RMDN model for testing. Here the recurrent layer has
    return_sequences=False and is stateful.

    :param hidden_states: dimension of the LSTM state.
    :param n_mixtures: number of output mixture densities.
    :param input_shape: the input shape like (1, c3d_encoding).
    :param summary: optional, whether or not to print summary.
    :return: a Keras model.
    """
    time_steps, _ = input_shape
    assert time_steps == 1, 'Input shape error, time step should be == 1 in test mode.'

    sequence_in = Input(batch_shape=(1,)+input_shape, name='input')

    state = LSTM(output_dim=hidden_states,
                 return_sequences=False,
                 stateful=True,
                 name='recurrent_module')(sequence_in)

    # Mixture Density Inference
    # mixture components weights
    weight = Dense(output_dim=n_mixtures * 1, name='output_weight')(state)
    weight = Activation('softmax')(weight)
    weight = Reshape(target_shape=(time_steps, n_mixtures, 1))(weight)

    # gaussian mean
    mu = Dense(output_dim=n_mixtures * 2, name='output_mean')(state)
    mu = Reshape(target_shape=(time_steps, n_mixtures, 2))(mu)
    mu = Activation('relu')(mu)  # this must become linear

    # variance
    sigma = Dense(output_dim=n_mixtures * 2, name='output_var')(state)
    sigma = Reshape(target_shape=(time_steps, n_mixtures, 2))(sigma)
    sigma = Lambda(lambda x: K.exp(x) + 1, output_shape=(time_steps, n_mixtures, 2))(sigma)

    # correlation
    ro = Dense(output_dim=n_mixtures, name='output_corr')(state)
    ro = Reshape(target_shape=(time_steps, n_mixtures, 1))(ro)
    ro = Activation('tanh')(ro)

    md = merge([weight, mu, sigma, ro], mode='concat', concat_axis=-1)

    model = Model(input=sequence_in, output=md)

    if summary:
        model.summary()

    return model


# helper entry point to test models
if __name__ == '__main__':
    from config import *

    # model = C3DEncoder(input_shape=(3, 16, 128, 171))
    model = RMDN_train(hidden_states=128, n_mixtures=C, input_shape=(T, encoding_dim))
    model.save_weights('prova.h5')
    model = RMDN_test(hidden_states=128, n_mixtures=C, input_shape=(1, encoding_dim))
    model.load_weights('prova.h5')
