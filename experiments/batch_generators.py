import numpy as np

from config import dreyeve_dir, dreyeve_train_seq, dreyeve_test_seq, frames_per_sequence
from random import choice
from os.path import join

from utils import palette
from computer_vision_utils.io_helper import read_image
from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.tensor_manipulation import resize_tensor

from time import time

import cv2


def dreyeve_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to load a batch of the dreyeve dataset
    :param batchsize: batchsize
    :param nb_frames: number of frames for each batch
    :param image_size: dimension of tensors
    :param mode: `train` or `test`
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth)
    :return: a tuple like: ([frames, of, semseg], y)
    """
    assert mode in ['train', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['sal', 'fix'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)

    if mode == 'train':
        sequences = dreyeve_train_seq
    elif mode == 'test':
        sequences = dreyeve_test_seq
    sequences = range(1, 20)  #todo remove this

    h, w = image_size

    X = np.zeros(shape=(batchsize, 3, nb_frames, h, w), dtype=np.float32)
    Y = np.zeros(shape=(batchsize, 1, nb_frames, h, w), dtype=np.float32)
    OF = np.zeros(shape=(batchsize, 3, nb_frames, h, w), dtype=np.float32)
    SEG = np.zeros(shape=(batchsize, 19, nb_frames, h, w), dtype=np.float32)
    for b in range(0, batchsize):
        # extract a random sequence
        seq = choice(sequences)

        x_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'frames')
        y_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency' if gt_type == 'sal' else 'saliency_fix')
        of_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'optical_flow')
        seg_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'semseg')

        # sample a frame
        start = np.random.randint(0, frames_per_sequence-nb_frames-1)  # -1 because we don't have OF for last frame
        for offset in range(0, nb_frames):
            X[b, :, offset, :, :] = read_image(join(x_dir, '{:06d}.jpg'.format(start + offset)),
                                               channels_first=True, resize_dim=image_size)
            Y[b, 0, offset, :, :] = read_image(join(y_dir, '{:06d}.png'.format(start + offset)),
                                               channels_first=True, color=False, resize_dim=image_size) / 255
            OF[b, :, offset, :, :] = read_image(join(of_dir, '{:06d}.png'.format(start + offset)),
                                                channels_first=True, resize_dim=image_size)
            SEG[b, :, offset, :, :] = resize_tensor(np.load(
                                                join(seg_dir, '{:06d}.npz'.format(start + offset)))['arr_0'][0],
                                                new_size=image_size)

    return [X, OF, SEG], Y


def visualize_batch(X, Y):
    """
    Helper function to visualize a batch
    :param X: input data: [frames, of, semseg]
    :param Y: saliency data
    """
    batchsize, _, frames_per_batch, h, w = X[0].shape

    X, OF, SEG = X
    for b in range(0, batchsize):
        for f in range(0, frames_per_batch):
            x = X[b, :, f, :, :].transpose(1, 2, 0)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

            of = OF[b, :, f, :, :].transpose(1, 2, 0)
            of = cv2.cvtColor(of, cv2.COLOR_RGB2BGR)

            # seg is different, we have to turn into colors
            seg = SEG[b, :, f, :, :]
            seg = palette[np.argmax(seg, axis=0).ravel()].reshape(h, w, 3)
            seg = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)

            # we have to turn y to 3 channels 255 for stitching
            y = Y[b, 0, f, :, :]
            y = (np.tile(y, (3, 1, 1))*255).transpose(1, 2, 0)

            # stitch and visualize
            stitch = stitch_together([x, of, seg, y], layout=(2, 2), resize_dim=(1080, 1920))
            cv2.imshow('batch viewer', stitch)
            cv2.waitKey(30)


def generate_dreyeve_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to generate a batch from the dreyeve dataset
    :param batchsize: batchsize
    :param nb_frames: number of frames for each batch
    :param image_size: dimension of tensors
    :param mode: `train` or `test`
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth)
    :return: a tuple like: ([frames, of, semseg], y)
    """
    while True:
        yield dreyeve_batch(batchsize=batchsize, nb_frames=nb_frames, image_size=image_size, mode=mode, gt_type=gt_type)


def test_load_batch():
    """
    Helper function, to load and visualize a dreyeve batch
    :return:
    """
    t = time()
    X, Y = dreyeve_batch(batchsize=4, nb_frames=16, image_size=(540, 960), mode='train')
    elapsed = time() - t

    print 'Batch loaded in {} seconds.'.format(elapsed)
    print 'X shape:{}'.format(X[0].shape)
    print 'OF shape:{}'.format(X[1].shape)
    print 'SEG shape:{}'.format(X[2].shape)
    print 'Y shape:{}'.format(Y.shape)

    visualize_batch(X, Y)

if __name__ == '__main__':
    test_load_batch()
