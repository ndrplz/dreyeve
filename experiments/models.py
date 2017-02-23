from keras.models import Model
from keras.layers import Input, Convolution3D, MaxPooling3D, Convolution2D, Reshape, Activation, merge, LeakyReLU, Lambda
from keras.utils.data_utils import get_file
from keras_dl_modules.custom_keras_extensions.layers import BilinearUpsampling


C3D_WEIGHTS_URL = 'http://imagelab.ing.unimore.it/files/c3d_weights/w_up2_conv4_new.h5'


def C3DEncoder(input_shape, pretrained=True, branch=''):
    c, fr, h, w = input_shape
    assert h % 8 == 0 and w % 8 == 0, 'I think input shape should be divisible by 8. Should it?'

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

    model_out = Reshape((512, h // 8, w // 8))(H)  # squeeze out temporal dimension

    model = Model(input=model_in, output=model_out, name='{}_c3d_encoder'.format(branch))

    if pretrained:
        weights_path = get_file('w_up2_conv4_new.h5', C3D_WEIGHTS_URL, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    return model


def SimpleSaliencyModel(input_shape, branch=''):
    c, fr, h, w = input_shape
    assert h % 8 == 0 and w % 8 == 0, 'I think input shape should be divisible by 8. Should it?'

    c3d_encoder = C3DEncoder(input_shape=(c, fr, h // 4, w // 4), branch=branch)

    # coarse + refinement
    ff_in = Input(shape=input_shape, name='input_ff')
    ff_last_frame = Lambda(lambda x: x[:, -1, :, :], output_shape=(c, 1, h, w))(ff_in)
    ff_last_frame = Reshape(target_shape=(c, h, w))(ff_last_frame)  # remove singleton dimension
    small_in = Input(shape=(c, fr, h // 4, w // 4), name='input_small')
    coarse_h = c3d_encoder(small_in)
    coarse_h = BilinearUpsampling(upsampling=32)(coarse_h)  # DVD: todo this 32x upsampling is temp

    fine_h = merge([coarse_h, ff_last_frame], mode='concat', concat_axis=1)
    fine_h = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(1, 1, 1, border_mode='same', init='he_normal')(fine_h)
    fine_out = Activation('sigmoid')(fine_h)

    # coarse on crop
    crop_in = Input(shape=(c, fr, h // 4, w // 4), name='input_crop')
    crop_h = c3d_encoder(crop_in)
    crop_h = BilinearUpsampling(upsampling=8)(crop_h)
    crop_h = Convolution2D(1, 1, 1, border_mode='same', init='he_normal')(crop_h)
    crop_out = Activation('sigmoid')(crop_h)

    model = Model(input=[ff_in, small_in, crop_in], output=[fine_out, crop_out],
                  name='{}_saliency_model'.format(branch))

    return model


def DreyeveNet(frames_per_seq, h, w):

    # get saliency branches
    im_net = SimpleSaliencyModel(input_shape=(3, frames_per_seq, h, w), branch='image')
    of_net = SimpleSaliencyModel(input_shape=(3, frames_per_seq, h, w), branch='optical_flow')
    seg_net = SimpleSaliencyModel(input_shape=(19, frames_per_seq, h, w), branch='segmentation')

    # define inputs todo add the full frame?
    X_ff = Input(shape=(3, frames_per_seq, h, w))
    X_small = Input(shape=(3, frames_per_seq, h // 4, w // 4))
    X_crop = Input(shape=(3, frames_per_seq, h // 4, w // 4))

    OF_ff = Input(shape=(3, frames_per_seq, h, w))
    OF_small = Input(shape=(3, frames_per_seq, h // 4, w // 4))
    OF_crop = Input(shape=(3, frames_per_seq, h // 4, w // 4))

    SEG_ff = Input(shape=(19, frames_per_seq, h, w))
    SEG_small = Input(shape=(19, frames_per_seq, h // 4, w // 4))
    SEG_crop = Input(shape=(19, frames_per_seq, h // 4, w // 4))

    x_pred_fine, x_pred_crop = im_net([X_ff, X_small, X_crop])
    of_pred_fine, of_pred_crop = of_net([OF_ff, OF_small, OF_crop])
    seg_pred_fine, seg_pred_crop = seg_net([SEG_ff, SEG_small, SEG_crop])

    fine_out = merge([x_pred_fine, of_pred_fine, seg_pred_fine], mode='sum')
    fine_out = Activation('sigmoid')(fine_out)

    crop_out = merge([x_pred_crop, of_pred_crop, seg_pred_crop], mode='sum')
    crop_out = Activation('sigmoid')(crop_out)

    model = Model(input=[X_ff, X_small, X_crop, OF_ff, OF_small, OF_crop, SEG_ff, SEG_small, SEG_crop],
                  output=[fine_out, crop_out], name='DreyeveNet')

    return model


if __name__ == '__main__':
    model = DreyeveNet(frames_per_seq=16, h=448, w=800)
    model.summary()
