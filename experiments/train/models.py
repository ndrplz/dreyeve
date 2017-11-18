import keras.backend as K

from keras.models import Model
from keras.layers import Input, Reshape, merge, Lambda, Activation, LeakyReLU
from keras.layers import Convolution3D, MaxPooling3D, Convolution2D
from keras.utils.data_utils import get_file

from config import simo_mode

from custom_layers import BilinearUpsampling


C3D_WEIGHTS_URL = 'http://imagelab.ing.unimore.it/files/c3d_weights/w_up2_conv4_new.h5'


def CoarseSaliencyModel(input_shape, pretrained, branch=''):
    """
    Function for constructing a CoarseSaliencyModel network, based on C3D.
    Used for coarse prediction in SaliencyBranch.

    :param input_shape: in the form (channels, frames, h, w).
    :param pretrained: Whether to initialize with weights pretrained on action recognition.
    :param branch: Name of the saliency branch (e.g. 'image' or 'optical_flow').
    :return: a Keras model.
    """
    c, fr, h, w = input_shape
    assert h % 8 == 0 and w % 8 == 0, 'I think input shape should be divisible by 8. Should it?'
    # if h % 8 != 0 and w % 8 != 0: # more polite just for debugging purpose
    #     print('Please consider that one of your input dimensions is not evenly divisible by 8.')

    # input_layers
    model_in = Input(shape=input_shape, name='input')

    # encoding net
    H = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1))(model_in)
    H = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(H)
    H = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(H)
    H = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(H)
    H = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(H)
    H = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(H)
    H = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1), border_mode='valid', name='pool4')(H)
    # DVD: once upon a time, this pooling had pool_size=(2, 2, 2) strides=(4, 2, 2)

    H = Reshape(target_shape=(512, h // 8, w // 8))(H)  # squeeze out temporal dimension

    model_out = BilinearUpsampling(upsampling=8, name='{}_8x_upsampling'.format(branch))(H)

    model = Model(input=model_in, output=model_out, name='{}_coarse_model'.format(branch))

    if pretrained:
        weights_path = get_file('w_up2_conv4_new.h5', C3D_WEIGHTS_URL, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    return model


def SaliencyBranch(input_shape, c3d_pretrained, branch=''):
    """
    Function for constructing a saliency model (coarse + fine). This will be a single branch
    of the final DreyeveNet.

    :param input_shape: in the form (channels, frames, h, w). h and w refer to the fullframe size.
    :param branch: Name of the saliency branch (e.g. 'image' or 'optical_flow').
    :return: a Keras model.
    """
    c, fr, h, w = input_shape
    # assert h % 32 == 0 and w % 32 == 0, 'I think input shape should be divisible by 32. Should it?'

    coarse_predictor = CoarseSaliencyModel(input_shape=(c, fr, h // 4, w // 4), pretrained=c3d_pretrained, branch=branch)

    ff_in = Input(shape=(c, 1, h, w), name='{}_input_ff'.format(branch))
    small_in = Input(shape=(c, fr, h // 4, w // 4), name='{}_input_small'.format(branch))
    crop_in = Input(shape=(c, fr, h // 4, w // 4), name='{}_input_crop'.format(branch))

    # coarse + refinement
    ff_last_frame = Reshape(target_shape=(c, h, w))(ff_in)  # remove singleton dimension
    coarse_h = coarse_predictor(small_in)
    coarse_h = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(coarse_h)
    coarse_h = BilinearUpsampling(upsampling=4, name='{}_4x_upsampling'.format(branch))(coarse_h)

    fine_h = merge([coarse_h, ff_last_frame], mode='concat', concat_axis=1, name='{}_full_frame_concat'.format(branch))
    fine_h = Convolution2D(32, 3, 3, border_mode='same', init='he_normal', name='{}_refine_conv1'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(16, 3, 3, border_mode='same', init='he_normal', name='{}_refine_conv2'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(8, 3, 3, border_mode='same', init='he_normal', name='{}_refine_conv3'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(1, 3, 3, border_mode='same', init='glorot_uniform', name='{}_refine_conv4'.format(branch))(fine_h)
    fine_out = Activation('relu')(fine_h)

    if simo_mode:
        # repeat fine_out tensor along axis=1, since we have two loss
        # DVD: this output shape is hardcoded, but should be fine
        fine_out = Lambda(lambda x: K.repeat_elements(x, rep=2, axis=1),
                          output_shape=(2, h, w), name='prediction_fine')(fine_out)
    else:
        fine_out = Activation('linear', name='prediction_fine')(fine_out)

    # coarse on crop
    crop_h = coarse_predictor(crop_in)
    crop_h = Convolution2D(1, 3, 3, border_mode='same', init='glorot_uniform', name='{}_crop_final_conv'.format(branch))(crop_h)
    crop_out = Activation('relu', name='prediction_crop')(crop_h)

    model = Model(input=[ff_in, small_in, crop_in], output=[fine_out, crop_out],
                  name='{}_saliency_branch'.format(branch))

    return model


def DreyeveNet(frames_per_seq, h, w):
    """
    Function for constructing the whole DreyeveNet.

    :param frames_per_seq: how many frames in each sequence.
    :param h: h (fullframe).
    :param w: w (fullframe).
    :return: a Keras model.
    """
    # get saliency branches
    im_net = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='image')
    of_net = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='optical_flow')
    seg_net = SaliencyBranch(input_shape=(19, frames_per_seq, h, w), c3d_pretrained=False, branch='segmentation')

    # define inputs
    X_ff = Input(shape=(3, 1, h, w), name='image_fullframe')
    X_small = Input(shape=(3, frames_per_seq, h // 4, w // 4), name='image_resized')
    X_crop = Input(shape=(3, frames_per_seq, h // 4, w // 4), name='image_cropped')

    OF_ff = Input(shape=(3, 1, h, w), name='flow_fullframe')
    OF_small = Input(shape=(3, frames_per_seq, h // 4, w // 4), name='flow_resized')
    OF_crop = Input(shape=(3, frames_per_seq, h // 4, w // 4), name='flow_cropped')

    SEG_ff = Input(shape=(19, 1, h, w), name='semseg_fullframe')
    SEG_small = Input(shape=(19, frames_per_seq, h // 4, w // 4), name='semseg_resized')
    SEG_crop = Input(shape=(19, frames_per_seq, h // 4, w // 4), name='semseg_cropped')

    x_pred_fine, x_pred_crop = im_net([X_ff, X_small, X_crop])
    of_pred_fine, of_pred_crop = of_net([OF_ff, OF_small, OF_crop])
    seg_pred_fine, seg_pred_crop = seg_net([SEG_ff, SEG_small, SEG_crop])

    fine_out = merge([x_pred_fine, of_pred_fine, seg_pred_fine], mode='sum', name='merge_fine_prediction')
    fine_out = Activation('relu', name='prediction_fine')(fine_out)

    crop_out = merge([x_pred_crop, of_pred_crop, seg_pred_crop], mode='sum', name='merge_crop_prediction')
    crop_out = Activation('relu', name='prediction_crop')(crop_out)

    model = Model(input=[X_ff, X_small, X_crop, OF_ff, OF_small, OF_crop, SEG_ff, SEG_small, SEG_crop],
                  output=[fine_out, crop_out], name='DreyeveNet')

    return model


# tester function
if __name__ == '__main__':
    model = SaliencyBranch(input_shape=(3, 16, 448, 448), c3d_pretrained=True, branch='image')
    model.summary()
