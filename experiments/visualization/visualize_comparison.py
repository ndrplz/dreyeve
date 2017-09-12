"""
Script that investigates frames where predictions are better than competitors with KL measures.
"""

import numpy as np
import cv2

import os
from os.path import join

from tqdm import tqdm

from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import read_image

from train.utils import read_lines_from_file

from utils import blend_map

dreyeve_root = '/majinbu/public/DREYEVE'
dreyeve_test_seq = range(38, 74+1)


def extract_most_competitive_frames(sequences, prediction_dir, competitors_dirs):
    """
    Function to extract frames where our model works better than competitors, in terms of KLD.

    :param sequences: list of sequences to consider (tipically, test sequences).
    :param prediction_dir: directory where our predictions stored.
    :param competitors_dirs: list of directories holding results from other competitors.
    :return: a list of tuples like (sequence, frame, gap) ordered by decreasing gap.
    """

    ret_list = []

    print 'Selecting best frames...'
    for seq in tqdm(sequences):
        my_kld_file = join(dreyeve_root, prediction_dir, '{:02d}'.format(seq), 'metrics', 'kld.txt')
        my_kld_list = read_lines_from_file(my_kld_file)[1:]  # remove head
        my_kld_list = [l[:-1].split(',') for l in my_kld_list]
        my_kld = np.array(my_kld_list, dtype=np.float32)

        comp_kld = []
        for c, comp in enumerate(competitors_dir):
            comp_kld_file = join(dreyeve_root, comp, '{:02d}'.format(seq), 'metrics', 'kld.txt')
            comp_kld_list = read_lines_from_file(comp_kld_file)[1:]  # remove head
            comp_kld_list = [l[:-1].split(',') for l in comp_kld_list]
            comp_kld.append(np.expand_dims(np.array(comp_kld_list, dtype=np.float32), axis=0))

        comp_kld = np.concatenate(comp_kld, axis=0)

        my_kld = my_kld[:, [0, 2]]
        comp_kld = comp_kld[:, :, [0, 2]]

        # index of frames where we perform better than all competitors (by means of kld)
        good_idx = list(np.where(np.all((my_kld[:, 1] < comp_kld[:, :, 1]), axis=0))[0])

        this_seq_list = [seq]*len(good_idx)
        frames_list = my_kld[good_idx, 0].tolist()
        gap = np.sum(np.square(my_kld[good_idx, 1] - comp_kld[:, good_idx, 1]), axis=0).tolist()

        ret_list += zip(this_seq_list, frames_list, gap)

    ret_list.sort(key=lambda x: x[2], reverse=True)
    return ret_list


def save_competitive_figures(out_dir, competitive_list, prediction_dir, competitors_dirs, resize_dim=(540, 960)):
    """
    Function that takes a list of competitive results and saves figures in a directory, for a future paper figure.

    :param out_dir: directory where to save figures.
    :param competitive_list: list of tuples like (sequence, frame, gap) ordered by decreasing gap.
    :param prediction_dir: directory where our predictions stored.
    :param competitors_dirs: list of directories holding results from other competitors.
    :param resize_dim: optional resize dim for output figures.
    """

    stitches_dir = join(out_dir, 'stitches')
    if not os.path.exists(stitches_dir):
        os.makedirs(stitches_dir)

    for i, competitive_result in enumerate(competitive_list):
        seq, frame, _ = competitive_result
        frame = int(frame)

        this_sample_dir = join(out_dir, '{:06d}'.format(i))
        if not os.path.exists(this_sample_dir):
            os.makedirs(this_sample_dir)

        # read frame
        im = read_image(join(dreyeve_root, 'DATA', '{:02d}'.format(seq), 'frames', '{:06d}.jpg'.format(frame)),
                           channels_first=False,
                           color_mode='BGR',
                           dtype=np.uint8,
                           resize_dim=resize_dim)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_000.jpg'.format(seq, frame)), im)

        # read gt and blend
        gt = read_image(join(dreyeve_root, 'DATA', '{:02d}'.format(seq), 'saliency', '{:06d}.png'.format(frame+1)),
                        channels_first=False,
                        color=False,
                        dtype=np.float32,
                        resize_dim=resize_dim)

        gt = blend_map(im, gt, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_001.jpg'.format(seq, frame)), gt)

        # read pred and blend
        my_pred = np.squeeze(cv2.imread(join(dreyeve_root, prediction_dir, '{:02d}'.format(seq),
                               'output', '{:06d}.png'.format(frame)), 0))
        my_pred = cv2.resize(my_pred, dsize=resize_dim[::-1])
        my_pred = blend_map(im, my_pred, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_002.jpg'.format(seq, frame)), my_pred)

        to_stitch = [im, gt, my_pred]

        for c, competitor in enumerate(competitors_dirs[:-1]):
            # read competitor result
            comp_pred = read_image(join(dreyeve_root, competitor, '{:02d}'.format(seq),
                                        'output', '{:06d}.png'.format(frame)),
                                   channels_first=False,
                                   color=False,
                                   dtype=np.float32,
                                   resize_dim=resize_dim)
            comp_pred = blend_map(im, comp_pred, factor=0.5)
            to_stitch.append(comp_pred)
            cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_{:03d}.jpg'.format(seq, frame, c+3)), comp_pred)

        stitch = stitch_together(to_stitch, layout=(1, len(to_stitch)))
        cv2.imwrite(join(stitches_dir, '{:06d}.jpg'.format(i)), stitch)


# entry point
if __name__ == '__main__':

    prediction_dir = 'PREDICTIONS/architecture7'
    competitors_dir = [
        # 'PREDICTIONS/architecture7',
        # 'PREDICTIONS_CENTRAL_GAUSSIAN',
        # 'PREDICTIONS_mathe',
        'PREDICTIONS_MLNET',
        'PREDICTIONS_RMDN',
        'PREDICTIONS_MEAN_TRAIN_GT',
        # 'PREDICTIONS_wang2015consistent',
        # 'PREDICTIONS_wang2015saliency'
    ]
    out_dir = 'competitive_figures'

    competitive_list = extract_most_competitive_frames(dreyeve_test_seq,
                                                       prediction_dir=prediction_dir, competitors_dirs=competitors_dir)

    save_competitive_figures(out_dir=out_dir, competitive_list=competitive_list,
                             prediction_dir=prediction_dir, competitors_dirs=competitors_dir)

