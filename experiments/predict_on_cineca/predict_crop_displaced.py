"""
This script predicts a dreyeve sequence with two displaced crops.
The model is trained only with central crop.
Will it generalise or learn the bias?
"""
import argparse
import os
from os.path import join

import numpy as np
import skimage.io as io
from computer_vision_utils.io_helper import read_image
from computer_vision_utils.tensor_manipulation import resize_tensor
from metrics.metrics import kld_numeric, cc_numeric
from skimage.transform import resize
from tqdm import tqdm
from train.models import DreyeveNet


def makedirs(dir_list):
    """
    Helper function to create a list of directories.

    :param dir_list: a list of directories to be created
    """

    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def translate(x, pixels, side):
    assert side in ['right', 'left']

    w = x.shape[-1]

    pad = x[..., (w - pixels):] if side == 'left' else x[..., :pixels]
    pad = pad[..., ::-1]

    if side == 'left':
        xt = np.roll(x, -pixels, axis=-1)
        xt[..., (w-pixels):] = pad
    else:
        xt = np.roll(x, pixels, axis=-1)
        xt[..., :pixels] = pad

    return xt


def load_dreyeve_cropped_samples(sequence_dir, sample, mean_dreyeve_image, frames_per_seq=16, h=448, w=448):
    """
    TODO.

    :param sequence_dir: string, sequence directory (e.g. 'Z:/DATA/04/').
    :param sample: int, sample to load in (15, 7499). N.B. this is the sample where prediction occurs!
    :param mean_dreyeve_image: mean dreyeve image, subtracted to each frame.
    :param frames_per_seq: number of temporal frames for each sample
    :param h: h
    :param w: w
    :return: a dreyeve_sample like I, OF, SEG
    """

    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    I_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    I_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    Il_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    Il_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    Il_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    Ir_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    Ir_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    Ir_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    OF_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    OF_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    OF_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    OFl_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    OFl_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    OFl_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    OFr_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    OFr_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    OFr_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    SEG_ff = np.zeros(shape=(1, 19, 1, h, w), dtype='float32')
    SEG_s = np.zeros(shape=(1, 19, frames_per_seq, h_s, w_s), dtype='float32')
    SEG_c = np.zeros(shape=(1, 19, frames_per_seq, h_c, w_c), dtype='float32')

    SEGl_ff = np.zeros(shape=(1, 19, 1, h, w), dtype='float32')
    SEGl_s = np.zeros(shape=(1, 19, frames_per_seq, h_s, w_s), dtype='float32')
    SEGl_c = np.zeros(shape=(1, 19, frames_per_seq, h_c, w_c), dtype='float32')

    SEGr_ff = np.zeros(shape=(1, 19, 1, h, w), dtype='float32')
    SEGr_s = np.zeros(shape=(1, 19, frames_per_seq, h_s, w_s), dtype='float32')
    SEGr_c = np.zeros(shape=(1, 19, frames_per_seq, h_c, w_c), dtype='float32')

    Y_sal = np.zeros(shape=(1, 1, h, w), dtype='float32')
    Y_fix = np.zeros(shape=(1, 1, h, w), dtype='float32')

    Yl_sal = np.zeros(shape=(1, 1, h, w), dtype='float32')
    Yl_fix = np.zeros(shape=(1, 1, h, w), dtype='float32')

    Yr_sal = np.zeros(shape=(1, 1, h, w), dtype='float32')
    Yr_fix = np.zeros(shape=(1, 1, h, w), dtype='float32')

    for fr in xrange(0, frames_per_seq):
        offset = sample - frames_per_seq + 1 + fr   # tricky

        # read image
        x = read_image(join(sequence_dir, 'frames', '{:06d}.jpg'.format(offset)), channels_first=True) \
            - mean_dreyeve_image
        xl = translate(x, pixels=500, side='left')
        xr = translate(x, pixels=500, side='right')
        I_s[0, :, fr, :, :] = resize_tensor(x, new_size=(h_s, w_s))
        Il_s[0, :, fr, :, :] = resize_tensor(xl, new_size=(h_s, w_s))
        Ir_s[0, :, fr, :, :] = resize_tensor(xr, new_size=(h_s, w_s))

        # read of
        of = read_image(join(sequence_dir, 'optical_flow', '{:06d}.png'.format(offset + 1)),
                        channels_first=True, resize_dim=(h, w))
        of -= np.mean(of, axis=(1, 2), keepdims=True)  # remove mean
        ofl = translate(of, pixels=500, side='left')
        ofr = translate(of, pixels=500, side='right')
        OF_s[0, :, fr, :, :] = resize_tensor(of, new_size=(h_s, w_s))
        OFl_s[0, :, fr, :, :] = resize_tensor(ofl, new_size=(h_s, w_s))
        OFr_s[0, :, fr, :, :] = resize_tensor(ofr, new_size=(h_s, w_s))

        # read semseg
        seg = resize_tensor(np.load(join(sequence_dir, 'semseg', '{:06d}.npz'.format(offset)))['arr_0'][0],
                            new_size=(h, w))
        segl = translate(seg, pixels=500, side='left')
        segr = translate(seg, pixels=500, side='right')

        SEG_s[0, :, fr, :, :] = resize_tensor(seg, new_size=(h_s, w_s))
        SEGl_s[0, :, fr, :, :] = resize_tensor(segl, new_size=(h_s, w_s))
        SEGr_s[0, :, fr, :, :] = resize_tensor(segr, new_size=(h_s, w_s))

    I_ff[0, :, 0, :, :] = resize_tensor(x, new_size=(h, w))
    Il_ff[0, :, 0, :, :] = resize_tensor(xl, new_size=(h, w))
    Ir_ff[0, :, 0, :, :] = resize_tensor(xr, new_size=(h, w))

    OF_ff[0, :, 0, :, :] = resize_tensor(of, new_size=(h, w))
    OFl_ff[0, :, 0, :, :] = resize_tensor(ofl, new_size=(h, w))
    OFr_ff[0, :, 0, :, :] = resize_tensor(ofr, new_size=(h, w))

    SEG_ff[0, :, 0, :, :] = resize_tensor(seg, new_size=(h, w))
    SEGl_ff[0, :, 0, :, :] = resize_tensor(segl, new_size=(h, w))
    SEGr_ff[0, :, 0, :, :] = resize_tensor(segr, new_size=(h, w))

    y_sal = read_image(join(sequence_dir, 'saliency', '{:06d}.png'.format(sample)), channels_first=False, color=False)
    yl_sal = translate(y_sal, pixels=500, side='left')
    yr_sal = translate(y_sal, pixels=500, side='right')
    Y_sal[0, 0] = resize_tensor(y_sal[np.newaxis, ...], new_size=(h, w))[0]
    Yl_sal[0, 0] = resize_tensor(yl_sal[np.newaxis, ...], new_size=(h, w))[0]
    Yr_sal[0, 0] = resize_tensor(yr_sal[np.newaxis, ...], new_size=(h, w))[0]

    y_fix = read_image(join(sequence_dir, 'saliency_fix', '{:06d}.png'.format(sample)), channels_first=False, color=False)
    yl_fix = translate(y_fix, pixels=500, side='left')
    yr_fix = translate(y_fix, pixels=500, side='right')
    Y_fix[0, 0] = resize_tensor(y_fix[np.newaxis, ...], new_size=(h, w))[0]
    Yl_fix[0, 0] = resize_tensor(yl_fix[np.newaxis, ...], new_size=(h, w))[0]
    Yr_fix[0, 0] = resize_tensor(yr_fix[np.newaxis, ...], new_size=(h, w))[0]

    X = [I_ff, I_s, I_c, OF_ff, OF_s, OF_c, SEG_ff, SEG_s, SEG_c]
    Xl = [Il_ff, Il_s, Il_c, OFl_ff, OFl_s, OFl_c, SEGl_ff, SEGl_s, SEGl_c]
    Xr = [Ir_ff, Ir_s, Ir_c, OFr_ff, OFr_s, OFr_c, SEGr_ff, SEGr_s, SEGr_c]
    GT = Y_sal, Y_fix
    GTl = Yl_sal, Yl_fix
    GTr = Yr_sal, Yr_fix
    return X, Xl, Xr, GT, GTl, GTr


import matplotlib.cm as cm
cm = cm.get_cmap('jet')


def save_blendmaps(path, tensors):
    X, Xl, Xr, Y, Yl, Yr, GT, GTl, GTr = map(np.squeeze, tensors)

    X, Xl, Xr = (X - np.min(X)), (Xl - np.min(Xl)), (Xr - np.min(Xr))
    X, Xl, Xr = (X / np.max(X)), (Xl / np.max(Xl)), (Xr / np.max(Xr))
    X, Xl, Xr = X.transpose(1, 2, 0), Xl.transpose(1, 2, 0), Xr.transpose(1, 2, 0)

    Y, Yl, Yr = (Y - np.min(Y)), (Yl - np.min(Yl)), (Yr - np.min(Yr))
    Y, Yl, Yr = (Y / np.max(Y)), (Yl / np.max(Yl)), (Yr / np.max(Yr))
    Y, Yl, Yr = map(cm, (Y, Yl, Yr))

    GT, GTl, GTr = (GT - np.min(GT)), (GTl - np.min(GTl)), (GTr - np.min(GTr))
    GT, GTl, GTr = (GT / np.max(GT)), (GTl / np.max(GTl)), (GTr / np.max(GTr))
    GT, GTl, GTr = map(cm, (GT, GTl, GTr))

    img_pred = np.concatenate((0.5 * Xl + 0.5 * Yl[:, :, :3],
                               0.5 * X + 0.5 * Y[:, :, :3],
                               0.5 * Xr + 0.5 * Yr[:, :, :3])
                              , axis=0)
    img_gt = np.concatenate((0.5 * Xl + 0.5 * GTl[:, :, :3],
                             0.5 * X + 0.5 * GT[:, :, :3],
                             0.5 * Xr + 0.5 * GTr[:, :, :3])
                            , axis=0)

    img = np.concatenate((img_pred, img_gt), axis=1)
    img = resize(img, (448 * 3, 448 * 4))

    io.imsave(path, img)


if __name__ == '__main__':

    frames_per_seq, h, w = 16, 448, 448
    verbose = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq")
    parser.add_argument("--pred_dir")
    args = parser.parse_args()

    assert args.seq is not None, 'Please provide a correct dreyeve sequence'
    assert args.pred_dir is not None, 'Please provide a correct pred_dir'

    dreyeve_dir = '/majinbu/public/DREYEVE/DATA'  # local

    # load mean dreyeve image
    mean_dreyeve_image = read_image(join(dreyeve_dir, 'dreyeve_mean_frame.png'), channels_first=True)

    # get the models
    dreyevenet_model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    dreyevenet_model.compile(optimizer='adam', loss='kld')  # do we need this?
    dreyevenet_model.load_weights('dreyevenet_model_central_crop.h5')  # load weights

    # set up pred directory
    image_pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'blend')
    makedirs([image_pred_dir])

    sequence_dir = join(dreyeve_dir, '{:02d}'.format(int(args.seq)))
    for sample in tqdm(range(15, 7500 - 1)):
        X, Xl, Xr, GT, GTl, GTr = load_dreyeve_cropped_samples(sequence_dir=sequence_dir,
                                                               sample=sample,
                                                               mean_dreyeve_image=mean_dreyeve_image,
                                                               frames_per_seq=frames_per_seq,
                                                               h=h,
                                                               w=w)
        GT_sal, GT_fix = GT
        GTl_sal, GTl_fix = GTl
        GTr_sal, GTr_fix = GTr

        Y = dreyevenet_model.predict(X)[0]
        Yl = dreyevenet_model.predict(Xl)[0]
        Yr = dreyevenet_model.predict(Xr)[0]

        # save model output
        save_blendmaps(join(image_pred_dir, '{:06d}.jpeg'.format(sample)),
                       (X[0], Xl[0], Xr[0], Y, Yl, Yr, GT_fix, GTl_fix, GTr_fix))

        # save some metrics
        with open(join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'kld.txt'), 'a') as metric_file:
            metric_file.write('{},{},{},{},{}\n'.format(sample,
                                                        kld_numeric(GT_sal, Y),
                                                        kld_numeric(GT_fix, Y),
                                                        kld_numeric(GTl_sal, Yl),
                                                        kld_numeric(GTl_fix, Yl),
                                                        kld_numeric(GTr_sal, Yr),
                                                        kld_numeric(GTr_fix, Yr)
                                                        ))
        with open(join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'cc.txt'), 'a') as metric_file:
            metric_file.write('{},{},{},{},{}\n'.format(sample,
                                                        cc_numeric(GT_sal, Y),
                                                        cc_numeric(GT_fix, Y),
                                                        cc_numeric(GTl_sal, Yl),
                                                        cc_numeric(GTl_fix, Yl),
                                                        cc_numeric(GTr_sal, Yr),
                                                        cc_numeric(GTr_fix, Yr)
                                                        ))
