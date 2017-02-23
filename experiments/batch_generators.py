import numpy as np

from config import dreyeve_dir, dreyeve_train_seq, dreyeve_test_seq, frames_per_sequence
from random import choice
from os.path import join

from utils import palette
from computer_vision_utils.io_helper import read_image
from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.tensor_manipulation import resize_tensor, crop_tensor

from time import time

import cv2


def dreyeve_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to load a batch of the dreyeve dataset

    :param batchsize: batchsize
    :param nb_frames: number of frames for each batch
    :param image_size: dimension of tensors, must satisfy (h,w) % 4 = (0,0)
    :param mode: `train` or `test`
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth)
    :return: a tuple like: [X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c], [Y, Y_c]
    """
    assert mode in ['train', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['sal', 'fix'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)

    if mode == 'train':
        sequences = dreyeve_train_seq
    elif mode == 'test':
        sequences = dreyeve_test_seq

    h, w = image_size
    assert (h % 4) == 0 and (w % 4) == 0, 'image size {} is not divisible by 4'.format(image_size)

    h_s = h_c = h // 4
    w_s = w_c = w // 4

    X = np.zeros(shape=(batchsize, 3, nb_frames, h, w), dtype=np.float32)
    X_s = np.zeros(shape=(batchsize, 3, nb_frames, h_s, w_s), dtype=np.float32)
    X_c = np.zeros(shape=(batchsize, 3, nb_frames, h_c, w_c), dtype=np.float32)
    OF = np.zeros(shape=(batchsize, 3, nb_frames, h, w), dtype=np.float32)
    OF_s = np.zeros(shape=(batchsize, 3, nb_frames, h_s, w_s), dtype=np.float32)
    OF_c = np.zeros(shape=(batchsize, 3, nb_frames, h_c, w_c), dtype=np.float32)
    SEG = np.zeros(shape=(batchsize, 19, nb_frames, h, w), dtype=np.float32)
    SEG_s = np.zeros(shape=(batchsize, 19, nb_frames, h_s, w_s), dtype=np.float32)
    SEG_c = np.zeros(shape=(batchsize, 19, nb_frames, h_c, w_c), dtype=np.float32)

    Y = np.zeros(shape=(batchsize, 1, h, w), dtype=np.float32)
    Y_c = np.zeros(shape=(batchsize, 1, h_c, w_c), dtype=np.float32)

    for b in range(0, batchsize):
        # extract a random sequence
        seq = choice(sequences)

        x_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'frames')
        y_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency' if gt_type == 'sal' else 'saliency_fix')
        of_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'optical_flow')
        seg_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'semseg')

        # evaluate random crop indexes
        hc1 = np.random.randint(0, h-h_c)
        hc2 = hc1 + h_c
        wc1 = np.random.randint(0, w-w_c)
        wc2 = wc1 + w_c

        # sample a frame
        start = np.random.randint(0, frames_per_sequence-nb_frames-1)  # -1 because we don't have OF for last frame
        for offset in range(0, nb_frames):
            # frame
            x = read_image(join(x_dir, '{:06d}.jpg'.format(start + offset)),
                           channels_first=True, resize_dim=image_size)
            X[b, :, offset, :, :] = x
            X_s[b, :, offset, :, :] = resize_tensor(x, new_size=(h_s, w_s))
            X_c[b, :, offset, :, :] = crop_tensor(x, indexes=(hc1, hc2, wc1, wc2))

            # optical flow
            of = read_image(join(of_dir, '{:06d}.png'.format(start + offset + 1)),
                            channels_first=True, resize_dim=image_size)
            OF[b, :, offset, :, :] = of
            OF_s[b, :, offset, :, :] = resize_tensor(of, new_size=(h_s, w_s))
            OF_c[b, :, offset, :, :] = crop_tensor(of, indexes=(hc1, hc2, wc1, wc2))

            # semantic segmentation
            seg = resize_tensor(np.load(join(seg_dir, '{:06d}.npz'.format(start + offset)))['arr_0'][0],
                                new_size=image_size)
            SEG[b, :, offset, :, :] = seg
            SEG_s[b, :, offset, :, :] = resize_tensor(seg, new_size=(h_s, w_s))
            SEG_c[b, :, offset, :, :] = crop_tensor(seg, indexes=(hc1, hc2, wc1, wc2))

        # saliency
        y = read_image(join(y_dir, '{:06d}.png'.format(start + nb_frames - 1)),
                       channels_first=True, color=False, resize_dim=image_size) / 255
        Y[b, 0, :, :] = y
        Y_c[b, 0, :, :] = crop_tensor(np.expand_dims(y, axis=0), indexes=(hc1, hc2, wc1, wc2))[0]

    return [X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c], [Y, Y_c]


def visualize_batch(X, Y):
    """
    Helper function to visualize a batch

    :param X: input data: [X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c]
    :param Y: saliency data like [Y, Y_c]
    """
    batchsize, _, frames_per_batch, h, w = X[0].shape
    batchsize, _, frames_per_batch, h_s, w_s = X[1].shape
    batchsize, _, frames_per_batch, h_c, w_c = X[2].shape

    X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c = X
    Y, Y_c = Y
    for b in range(0, batchsize):
        for f in range(0, frames_per_batch):
            # FULL FRAME SECTION -----
            x = X[b, :, f, :, :].transpose(1, 2, 0)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

            of = OF[b, :, f, :, :].transpose(1, 2, 0)
            of = cv2.cvtColor(of, cv2.COLOR_RGB2BGR)

            # seg is different, we have to turn into colors
            seg = SEG[b, :, f, :, :]
            seg = palette[np.argmax(seg, axis=0).ravel()].reshape(h, w, 3)
            seg = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)

            # we have to turn y to 3 channels 255 for stitching
            y = Y[b, 0, :, :]
            y = (np.tile(y, (3, 1, 1))*255).transpose(1, 2, 0)

            # stitch and visualize
            stitch_ff = stitch_together([x, of, seg, y], layout=(2, 2), resize_dim=(540, 960))

            # CROPPED FRAME SECTION -----
            x_c = X_c[b, :, f, :, :].transpose(1, 2, 0)
            x_c = cv2.cvtColor(x_c, cv2.COLOR_RGB2BGR)

            of_c = OF_c[b, :, f, :, :].transpose(1, 2, 0)
            of_c = cv2.cvtColor(of_c, cv2.COLOR_RGB2BGR)

            # seg is different, we have to turn into colors
            seg_c = SEG_c[b, :, f, :, :]
            seg_c = palette[np.argmax(seg_c, axis=0).ravel()].reshape(h_c, w_c, 3)
            seg_c = cv2.cvtColor(seg_c, cv2.COLOR_RGB2BGR)

            # we have to turn y to 3 channels 255 for stitching
            y_c = Y_c[b, 0, :, :]
            y_c = (np.tile(y_c, (3, 1, 1)) * 255).transpose(1, 2, 0)

            # stitch and visualize
            stitch_c = stitch_together([x_c, of_c, seg_c, y_c], layout=(2, 2), resize_dim=(540, 960))

            # SMALL FRAME SECTION -----
            x_s = X_s[b, :, f, :, :].transpose(1, 2, 0)
            x_s = cv2.cvtColor(x_s, cv2.COLOR_RGB2BGR)

            of_s = OF_s[b, :, f, :, :].transpose(1, 2, 0)
            of_s = cv2.cvtColor(of_s, cv2.COLOR_RGB2BGR)

            # seg is different, we have to turn into colors
            seg_s = SEG_s[b, :, f, :, :]
            seg_s = palette[np.argmax(seg_s, axis=0).ravel()].reshape(h_s, w_s, 3)
            seg_s = cv2.cvtColor(seg_s, cv2.COLOR_RGB2BGR)

            # we have to turn y to 3 channels 255 for stitching
            # also, we resize it to small (just for visualize it)
            y_s = cv2.resize(Y[b, 0, :, :], dsize=(h_s, w_s)[::-1])
            y_s = (np.tile(y_s, (3, 1, 1)) * 255).transpose(1, 2, 0)

            # stitch and visualize
            stitch_s = stitch_together([x_s, of_s, seg_s, y_s], layout=(2, 2), resize_dim=(540, 960))

            # stitch the stitchs D=
            final_stitch = stitch_together([stitch_ff, stitch_s, stitch_c], layout=(1,3), resize_dim=(810, 1440))
            cv2.imshow('Batch_viewer', final_stitch)
            cv2.waitKey(30)


def generate_dreyeve_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to generate a batch from the dreyeve dataset

    :param batchsize: batchsize
    :param nb_frames: number of frames for each batch
    :param image_size: dimension of tensors
    :param mode: `train` or `test`
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth)
    :return: a tuple like: ([X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c], [Y, Y_c])
    """
    while True:
        yield dreyeve_batch(batchsize=batchsize, nb_frames=nb_frames, image_size=image_size, mode=mode, gt_type=gt_type)


def test_load_batch():
    """
    Helper function, to load and visualize a dreyeve batch

    :return:
    """
    t = time()
    X, Y = dreyeve_batch(batchsize=8, nb_frames=16, image_size=(448, 800), mode='test')
    elapsed = time() - t

    print 'Batch loaded in {} seconds.'.format(elapsed)
    print 'X shape:{}'.format(X[0].shape)
    print 'X_s shape:{}'.format(X[1].shape)
    print 'X_c shape:{}'.format(X[2].shape)
    print 'OF shape:{}'.format(X[3].shape)
    print 'OF_s shape:{}'.format(X[4].shape)
    print 'OF_c shape:{}'.format(X[5].shape)
    print 'SEG shape:{}'.format(X[6].shape)
    print 'SEG_s shape:{}'.format(X[7].shape)
    print 'SEG_c shape:{}'.format(X[8].shape)
    print 'Y shape:{}'.format(Y[0].shape)
    print 'Y_c shape:{}'.format(Y[1].shape)

    visualize_batch(X, Y)

if __name__ == '__main__':
    test_load_batch()
