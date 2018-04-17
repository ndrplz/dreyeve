import numpy as np
import cv2

import argparse

import os
from tqdm import tqdm
from os.path import join

from train.models import DreyeveNet
from computer_vision_utils.io_helper import read_image, normalize
from computer_vision_utils.tensor_manipulation import resize_tensor
from computer_vision_utils.stitching import stitch_together

from train.utils import seg_to_colormap
from metrics.metrics import kld_numeric, cc_numeric


def makedirs(dir_list):
    """
    Helper function to create a list of directories.

    :param dir_list: a list of directories to be created
    """

    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def load_dreyeve_sample(sequence_dir, sample, mean_dreyeve_image, frames_per_seq=16, h=448, w=448, ):
    """
    Function to load a dreyeve_sample.

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
    OF_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    OF_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    OF_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')
    SEG_ff = np.zeros(shape=(1, 19, 1, h, w), dtype='float32')
    SEG_s = np.zeros(shape=(1, 19, frames_per_seq, h_s, w_s), dtype='float32')
    SEG_c = np.zeros(shape=(1, 19, frames_per_seq, h_c, w_c), dtype='float32')

    Y_sal = np.zeros(shape=(1, 1, h, w), dtype='float32')
    Y_fix = np.zeros(shape=(1, 1, h, w), dtype='float32')

    for fr in xrange(0, frames_per_seq):
        offset = sample - frames_per_seq + 1 + fr   # tricky

        # read image
        x = read_image(join(sequence_dir, 'frames', '{:06d}.jpg'.format(offset)),
                       channels_first=True, resize_dim=(h, w)) - mean_dreyeve_image
        I_s[0, :, fr, :, :] = resize_tensor(x, new_size=(h_s, w_s))

        # read of
        of = read_image(join(sequence_dir, 'optical_flow', '{:06d}.png'.format(offset + 1)),
                        channels_first=True, resize_dim=(h, w))
        of -= np.mean(of, axis=(1, 2), keepdims=True)  # remove mean
        OF_s[0, :, fr, :, :] = resize_tensor(of, new_size=(h_s, w_s))

        # read semseg
        seg = resize_tensor(np.load(join(sequence_dir, 'semseg', '{:06d}.npz'.format(offset)))['arr_0'][0],
                            new_size=(h, w))
        SEG_s[0, :, fr, :, :] = resize_tensor(seg, new_size=(h_s, w_s))

    I_ff[0, :, 0, :, :] = x
    OF_ff[0, :, 0, :, :] = of
    SEG_ff[0, :, 0, :, :] = seg

    Y_sal[0, 0] = read_image(join(sequence_dir, 'saliency', '{:06d}.png'.format(sample)), channels_first=False,
                             color=False, resize_dim=(h, w))
    Y_fix[0, 0] = read_image(join(sequence_dir, 'saliency_fix', '{:06d}.png'.format(sample)), channels_first=False,
                             color=False, resize_dim=(h, w))

    return [I_ff, I_s, I_c, OF_ff, OF_s, OF_c, SEG_ff, SEG_s, SEG_c], [Y_sal, Y_fix]


if __name__ == '__main__':

    frames_per_seq, h, w = 16, 448, 448
    verbose = False

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int)
    parser.add_argument("--pred_dir", type=str)
    args = parser.parse_args()

    assert args.seq is not None, 'Please provide a correct dreyeve sequence'
    assert args.pred_dir is not None, 'Please provide a correct pred_dir'

    dreyeve_dir = '/tmp/DREYEVE_DATA'  # aimagelab-local
    #dreyeve_dir = '/nas/majinbu/DREYEVE/DATA'  # aimagelab-majinbu
    # dreyeve_dir = '/gpfs/work/IscrC_DeepVD/dabati/DREYEVE/data/'  # cineca

    # load mean dreyeve image
    mean_dreyeve_image = read_image(join(dreyeve_dir, 'dreyeve_mean_frame.png'),
                                    channels_first=True, resize_dim=(h, w))

    # get the models
    dreyevenet_model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    dreyevenet_model.compile(optimizer='adam', loss='kld')  # do we need this?
    dreyevenet_model.load_weights('dreyevenet_model_central_crop.h5')  # load weights

    image_branch = [l for l in dreyevenet_model.layers if l.name == 'image_saliency_branch'][0]
    flow_branch = [l for l in dreyevenet_model.layers if l.name == 'optical_flow_saliency_branch'][0]
    semseg_branch = [l for l in dreyevenet_model.layers if l.name == 'segmentation_saliency_branch'][0]

    # set up some directories
    dreyevenet_pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'dreyeveNet')
    image_pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'image_branch')
    flow_pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'flow_branch')
    semseg_pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'semseg_branch')
    makedirs([dreyevenet_pred_dir, image_pred_dir, flow_pred_dir, semseg_pred_dir])

    sequence_dir = join(dreyeve_dir, '{:02d}'.format(int(args.seq)))
    for sample in tqdm(range(15, 7500 - 1)):
        from time import time
        t = time()
        X, GT = load_dreyeve_sample(sequence_dir=sequence_dir, sample=sample, mean_dreyeve_image=mean_dreyeve_image,
                                frames_per_seq=frames_per_seq, h=h, w=w)
        print(time() - t)
        GT_sal, GT_fix = GT

        Y_dreyevenet = dreyevenet_model.predict(X)[0]  # get only [fine_out][remove batch]
        Y_image = image_branch.predict(X[:3])[0]  # predict on image
        Y_flow = flow_branch.predict(X[3:6])[0]  # predict on optical flow
        Y_semseg = semseg_branch.predict(X[6:])[0]  # predict on segmentation

        # save model output
        np.savez_compressed(join(dreyevenet_pred_dir, '{:06d}'.format(sample)), Y_dreyevenet)
        np.savez_compressed(join(image_pred_dir, '{:06d}'.format(sample)), Y_image)
        np.savez_compressed(join(flow_pred_dir, '{:06d}'.format(sample)), Y_flow)
        np.savez_compressed(join(semseg_pred_dir, '{:06d}'.format(sample)), Y_semseg)

        # save some metrics
        with open(join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'kld.txt'), 'a') as metric_file:
            metric_file.write('{},{},{},{},{},{},{},{},{}\n'.format(sample,
                                                                    kld_numeric(GT_sal, Y_dreyevenet),
                                                                    kld_numeric(GT_fix, Y_dreyevenet),
                                                                    kld_numeric(GT_sal, Y_image),
                                                                    kld_numeric(GT_fix, Y_image),
                                                                    kld_numeric(GT_sal, Y_flow),
                                                                    kld_numeric(GT_fix, Y_flow),
                                                                    kld_numeric(GT_sal, Y_semseg),
                                                                    kld_numeric(GT_fix, Y_semseg),
                                                                    ))
        with open(join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'cc.txt'), 'a') as metric_file:
            metric_file.write('{},{},{},{},{},{},{},{},{}\n'.format(sample,
                                                                    cc_numeric(GT_sal, Y_dreyevenet),
                                                                    cc_numeric(GT_fix, Y_dreyevenet),
                                                                    cc_numeric(GT_sal, Y_image),
                                                                    cc_numeric(GT_fix, Y_image),
                                                                    cc_numeric(GT_sal, Y_flow),
                                                                    cc_numeric(GT_fix, Y_flow),
                                                                    cc_numeric(GT_sal, Y_semseg),
                                                                    cc_numeric(GT_fix, Y_semseg),
                                                                    ))

        if verbose:
            # visualization
            x_stitch = stitch_together([normalize(X[0][0, :, 0, :, :].transpose(1, 2, 0)),
                                        normalize(X[3][0, :, 0, :, :].transpose(1, 2, 0)),
                                        normalize(seg_to_colormap(np.argmax(X[6][0, :, 0, :, :], axis=0),
                                                                  channels_first=False))],
                                       layout=(3, 1), resize_dim=(720, 720))

            y_stitch = stitch_together([np.tile(normalize(Y_image[0].transpose(1, 2, 0)), reps=(1, 1, 3)),
                                        np.tile(normalize(Y_flow[0].transpose(1, 2, 0)), reps=(1, 1, 3)),
                                        np.tile(normalize(Y_semseg[0].transpose(1, 2, 0)), reps=(1, 1, 3))],
                                       layout=(3, 1), resize_dim=(720, 720))

            y_tot = np.tile(normalize(resize_tensor(Y_dreyevenet[0], new_size=(720, 720)).transpose(1, 2, 0)),
                            reps=(1, 1, 3))

            cv2.imshow('prediction', stitch_together([x_stitch, y_stitch, y_tot], layout=(1, 3),
                                                     resize_dim=(500, 1500)))
            cv2.waitKey(1)
