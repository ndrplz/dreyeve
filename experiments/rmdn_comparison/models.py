"""
This file holds models for comparison with RMDN.
"""

from keras.models import Model
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten
from keras.layers import LSTM, Dense, TimeDistributedDense, Reshape, merge, Activation
from keras.utils.data_utils import get_file

from objectives import MDN_neg_log_likelyhood


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

    time_steps, _ = input_shape

    sequence_in = Input(shape=input_shape, name='input')

    state = LSTM(output_dim=hidden_states, return_sequences=True, name='recurrent_module')(sequence_in)

    # Mixture Density Inference
    weight = TimeDistributedDense(output_dim=n_mixtures * 1, name='output_weight')(state)
    weight = Reshape(target_shape=(time_steps, n_mixtures, 1))(weight)
    weight = Activation('linear')(weight)

    mean = TimeDistributedDense(output_dim=n_mixtures * 2, name='output_mean')(state)
    mean = Reshape(target_shape=(time_steps * n_mixtures, 2))(mean)
    mean = Activation('softmax')(mean)
    mean = Reshape(target_shape=(time_steps, n_mixtures, 2))(mean)

    var = TimeDistributedDense(output_dim=n_mixtures * 2, name='output_var')(state)
    var = Reshape(target_shape=(time_steps, n_mixtures, 2))(var)
    var = Activation('relu')(var)  # TODO this activation must become exp

    corr = TimeDistributedDense(output_dim=n_mixtures * 2, name='output_corr')(state)
    corr = Reshape(target_shape=(time_steps, n_mixtures, 2))(corr)
    corr = Activation('tanh')(corr)

    md = merge([mean, weight, var, corr], mode='concat', concat_axis=-1)

    model = Model(input=sequence_in, output=md)

    if summary:
        model.summary()

    return model


def RMDN_test(hidden_states, n_mixtures, input_shape, summary=True):

    time_steps, _ = input_shape
    assert time_steps == 1, 'Input shape error, time step should be == 1 in test mode.'

    sequence_in = Input(batch_shape=(1,)+input_shape, name='input')

    state = LSTM(output_dim=hidden_states,
                 return_sequences=False,
                 stateful=True,
                 name='recurrent_module')(sequence_in)

    # Mixture Density Inference
    weight = Dense(output_dim=n_mixtures * 1, name='output_weight')(state)
    weight = Reshape(target_shape=(n_mixtures, 1))(weight)
    weight = Activation('linear')(weight)

    mean = Dense(output_dim=n_mixtures * 2, name='output_mean')(state)
    mean = Reshape(target_shape=(n_mixtures, 2))(mean)
    mean = Activation('softmax')(mean)

    var = Dense(output_dim=n_mixtures * 2, name='output_var')(state)
    var = Reshape(target_shape=(n_mixtures, 2))(var)
    var = Activation('relu')(var)  # TODO this activation must become exp

    corr = Dense(output_dim=n_mixtures * 2, name='output_corr')(state)
    corr = Reshape(target_shape=(n_mixtures, 2))(corr)
    corr = Activation('tanh')(corr)

    md = merge([mean, weight, var, corr], mode='concat', concat_axis=-1)

    model = Model(input=sequence_in, output=md)

    if summary:
        model.summary()

    return model


def RMDN(hidden_states, n_mixtures, input_shape, mode, summary=True):

    assert mode in ['train', 'test'], 'Unknown mode {}'.format(mode)

    if mode == 'train':
        model = RMDN_train(hidden_states, n_mixtures, input_shape, summary)
    else:
        model = RMDN_test(hidden_states, n_mixtures, input_shape, summary)

    return model


# helper entry point to test models
if __name__ == '__main__':

    # model = C3DEncoder(input_shape=(3, 16, 112, 112))
    model = RMDN_train(hidden_states=128, n_mixtures=20, input_shape=(30, 50176))
    model.compile(optimizer='adam', loss=MDN_neg_log_likelyhood)