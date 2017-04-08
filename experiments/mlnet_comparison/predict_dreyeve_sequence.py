import numpy as np
import cv2

import argparse

import os
from tqdm import tqdm
from os.path import join

from model import ml_net_model
from utils import preprocess_images, postprocess_predictions
from computer_vision_utils.io_helper import normalize
from computer_vision_utils.stitching import stitch_together

from config import shape_r, shape_c


def makedirs(dir_list):
    """
    Helper function to create a list of directories.

    :param dir_list: a list of directories to be created
    """

    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def load_dreyeve_sample(sequence_dir, sample, shape_r, shape_c):
    """
    Function to load a dreyeve sample.

    :param sequence_dir: the directory of the sequence we want to sample from.
    :param sample: the number of the sample.
    :param shape_r: rows of the image to predict.
    :param shape_c: columns of the image to predict.
    :return: a ndarray having shape (1, 3, shape_r, shape_w)
    """

    filename = join(sequence_dir, 'frames', '{:06d}.jpg'.format(sample))
    X = preprocess_images([filename], shape_r, shape_c)

    return X


if __name__ == '__main__':

    verbose = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq")
    parser.add_argument("--pred_dir")
    args = parser.parse_args()

    assert args.seq is not None, 'Please provide a correct dreyeve sequence'
    assert args.pred_dir is not None, 'Please provide a correct pred_dir'

    dreyeve_dir = 'Z:/DATA'  # local
    # dreyeve_dir = '/gpfs/work/IscrC_DeepVD/dabati/DREYEVE/data/'  # cineca

    # get the model
    model = ml_net_model(img_rows=shape_r, img_cols=shape_c)
    model.compile(optimizer='adam', loss='kld')  # do we need this?
    model.load_weights('weights.mlnet.07-0.0193.pkl')  # load weights

    # set up some directories
    pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'output')
    makedirs([pred_dir])

    sequence_dir = join(dreyeve_dir, '{:02d}'.format(int(args.seq)))
    for sample in tqdm(range(15, 7500 - 1)):
        X = load_dreyeve_sample(sequence_dir=sequence_dir, sample=sample, shape_c=shape_c, shape_r=shape_r)

        # predict sample
        P = model.predict(X)
        P = np.squeeze(P)

        # save model output
        P = postprocess_predictions(P, shape_r, shape_c)
        cv2.imwrite(join(pred_dir, '{:06d}.png'.format(sample)), P)

        if verbose:
            # visualization
            x_img = X[0].transpose(1, 2, 0)
            p_img = cv2.cvtColor(P, cv2.COLOR_GRAY2BGR)
            stitch = stitch_together([normalize(x_img), p_img], layout=(1, 2))

            cv2.imshow('predition', stitch)
            cv2.waitKey(1)


