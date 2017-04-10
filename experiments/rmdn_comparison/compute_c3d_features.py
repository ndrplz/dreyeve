"""
This script is used to compute C3D features over a complete Dr(eye)ve sequence.
Usage:
    python compute_c3d_features.py --seq <seq>
"""

import numpy as np
import argparse
import os

from os.path import join
from tqdm import tqdm

from rmdn_comparison.models import C3DEncoder

from computer_vision_utils.io_helper import read_image

from config import DREYEVE_ROOT


def load_sample_for_encoding(frames_dir, sample_number, temporal_window, img_size):
    """
    Loads a sample to be encoded by C3D.

    :param frames_dir: directory where the frames are.
    :param sample_number: the frame index.
    :param temporal_window: the temporal window we consider (16 frames).
    :param img_size: the size of images C3D has to be fed with.
    :return: a ndarray having shape (1, 3, temporal_window, img_size[0], img_size[1].
    """

    h, w = img_size
    sample = np.zeros(shape=(3, temporal_window, h, w), dtype=np.float32)

    # if sample_number==15 --> [0,15]
    start = sample_number - temporal_window + 1
    stop = sample_number + 1
    for f in xrange(start, stop):
        img = read_image(img_path=join(frames_dir, '{:06d}.jpg'.format(f)),
                         channels_first=True,
                         resize_dim=(h, w))

        t_idx = f - start
        sample[:, t_idx, :, :] = img

    # add batch dimension
    sample = np.expand_dims(sample, axis=0)

    return sample

# main script
if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq")
    args = parser.parse_args()

    assert args.seq is not None, 'Please provide a correct dreyeve sequence'

    # some parameters
    c, f, h, w = (3, 16, 128, 171)

    dreyeve_dir = DREYEVE_ROOT  # local

    # get the model
    model = C3DEncoder(input_shape=(c, f, h, w))

    # set up some directories
    sequence_dir = join(dreyeve_dir, '{:02d}'.format(int(args.seq)))
    frames_dir = join(sequence_dir, 'frames')
    encoding_dir = join(sequence_dir, 'c3d_encodings')
    if not os.path.exists(encoding_dir):
        os.makedirs(encoding_dir)

    # begin 
    for frame in tqdm(xrange(15, 7500)):
        sample = load_sample_for_encoding(frames_dir, sample_number=frame, temporal_window=f, img_size=(h, w))
        encoding = model.predict(sample)
        encoding = np.squeeze(encoding)  # remove batch singleton dimension

        np.savez_compressed(join(encoding_dir, '{:06d}.npz'.format(frame)), encoding)
