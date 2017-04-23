"""
This script computes, given a sequence, all predictions using RMDN model.

Usage:
python predict_dreyeve_sequence.py --seq <sequence_num> --pred_dir <output_dir>
"""
import numpy as np
import cv2

import argparse

import os
from tqdm import tqdm
from os.path import join

from models import RMDN_test
from computer_vision_utils.io_helper import normalize

from config import hidden_states, C, encoding_dim, h, w, DREYEVE_ROOT
from utils import gmm_to_probability_map


def makedirs(dir_list):
    """
    Helper function to create a list of directories.

    :param dir_list: a list of directories to be created
    """

    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def load_dreyeve_sample(sequence_dir, sample):
    """
    Function to load a c3d encoding, given the sequence and the sample number.

    :param sequence_dir: the directory of the sequence we want to sample from.
    :param sample: the number of the sample.
    :return: a ndarray having shape (1, 3, shape_r, shape_w)
    """

    filename = join(sequence_dir, 'c3d_encodings', '{:06d}.npz'.format(sample))
    encoding = np.load(filename)['arr_0']

    encoding = np.expand_dims(encoding, axis=0)  # temporal dimension
    encoding = np.expand_dims(encoding, axis=0)  # batch dimension

    return encoding


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq")
    parser.add_argument("--pred_dir")
    args = parser.parse_args()

    assert args.seq is not None, 'Please provide a correct dreyeve sequence'
    assert args.pred_dir is not None, 'Please provide a correct pred_dir'

    # get the model
    model = RMDN_test(hidden_states=hidden_states, n_mixtures=C, input_shape=(1, encoding_dim))
    model.load_weights('bazzani.h5')

    # set up some directories
    pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'output')
    makedirs([pred_dir])

    sequence_dir = join(DREYEVE_ROOT, '{:02d}'.format(int(args.seq)))
    for sample in tqdm(range(15, 7500)):
        X = load_dreyeve_sample(sequence_dir=sequence_dir, sample=sample)

        # predict sample
        P = model.predict(X)
        P_map = gmm_to_probability_map(P[0, 0], image_size=(h, w))

        # save model output
        cv2.imwrite(join(pred_dir, '{:06d}.png'.format(sample)), normalize(P_map))
