"""
Script that investigates frames where global predictions are 
better than single branch predictions with KL measures.
"""
from __future__ import print_function
import os
import cv2
import numpy as np
from os.path import join
from tqdm import tqdm
from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import read_image
from train.utils import read_lines_from_file
from visualization.utils import blend_map


dreyeve_root = '/majinbu/public/DREYEVE'
dreyeve_test_seq = range(38, 74+1)


def extract_most_competitive_frames(sequences, prediction_dir):
    """
    Function to extract frames where our model works better than single branches, in terms of KLD.

    :param sequences: list of sequences to consider (tipically, test sequences).
    :param prediction_dir: directory where our predictions stored.
    :return: a list of tuples like (sequence, frame, gap) ordered by decreasing gap.
    """

    ret_list = []

    print('Selecting best frames...')
    for seq in tqdm(sequences):
        kld_file = join(dreyeve_root, prediction_dir, '{:02d}'.format(seq), 'metrics', 'kld.txt')
        kld_list = read_lines_from_file(kld_file)[1:]  # remove head
        kld_list = [l[:-1].split(',') for l in kld_list]
        kld = np.array(kld_list, dtype=np.float32)

        my_kld = kld[:, [0, 5]]
        comp_kld = kld[:, [0, 6, 7, 8]]

        # index of frames where we perform better than all competitors (by means of kld)
        good_idx = list(np.where(np.all((np.tile(my_kld[:, 1:], reps=(1, 3)) < comp_kld[:, 1:]), axis=1))[0])

        this_seq_list = [seq]*len(good_idx)
        frames_list = my_kld[good_idx, 0].tolist()
        gap = np.sum(np.square(np.stack([my_kld[good_idx, 1]]*3, axis=-1) - comp_kld[good_idx, 1:]), axis=-1).tolist()

        ret_list += zip(this_seq_list, frames_list, gap)

    ret_list.sort(key=lambda x: x[2], reverse=True)
    return ret_list


def save_competitive_figures(out_dir, competitive_list, prediction_dir, resize_dim=(540, 960)):
    """
    Function that takes a list of competitive results and saves figures in a directory, for a future paper figure.

    :param out_dir: directory where to save figures.
    :param competitive_list: list of tuples like (sequence, frame, gap) ordered by decreasing gap.
    :param prediction_dir: directory where our predictions stored.
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
        gt = read_image(join(dreyeve_root, 'DATA', '{:02d}'.format(seq), 'saliency_fix', '{:06d}.png'.format(frame+1)),
                        channels_first=False,
                        color=False,
                        dtype=np.float32,
                        resize_dim=resize_dim)

        gt = blend_map(im, gt, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_001.jpg'.format(seq, frame)), gt)

        # read pred and blend
        my_pred = np.squeeze(np.load(join(dreyeve_root, prediction_dir, '{:02d}'.format(seq),
                                          'dreyeveNet', '{:06d}.npz'.format(frame)))['arr_0'])
        my_pred = cv2.resize(my_pred, dsize=resize_dim[::-1])
        my_pred = blend_map(im, my_pred, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_002.jpg'.format(seq, frame)), my_pred)

        to_stitch = [im, gt, my_pred]

        for c, competitor in enumerate(['image_branch', 'flow_branch', 'semseg_branch']):
            # read competitor result
            comp_pred = np.squeeze(np.load(join(dreyeve_root, prediction_dir, '{:02d}'.format(seq),
                                                competitor, '{:06d}.npz'.format(frame)))['arr_0'])
            comp_pred = cv2.resize(comp_pred, dsize=resize_dim[::-1])

            comp_pred = blend_map(im, comp_pred, factor=0.5)
            to_stitch.append(comp_pred)
            cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_{:03d}.jpg'.format(seq, frame, c+3)), comp_pred)

        stitch = stitch_together(to_stitch, layout=(1, len(to_stitch)))
        cv2.imwrite(join(stitches_dir, '{:06d}.jpg'.format(i)), stitch)


# entry point
if __name__ == '__main__':

    prediction_dir = 'PREDICTIONS_2017'

    out_dir = 'competitive_figures_ablation'

    competitive_list = extract_most_competitive_frames(dreyeve_test_seq, prediction_dir=prediction_dir)

    save_competitive_figures(out_dir=out_dir, competitive_list=competitive_list,
                             prediction_dir=prediction_dir)
