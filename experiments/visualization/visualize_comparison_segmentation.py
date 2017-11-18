"""
Script that investigates frames in which segmentation performs better than other branches.
"""

import numpy as np
import cv2

import os
from os.path import join

from tqdm import tqdm

from computer_vision_utils.io_helper import read_image

from train.utils import read_lines_from_file

from utils import blend_map, seg_to_rgb

dreyeve_root = '/majinbu/public/DREYEVE'
dreyeve_test_seq = range(38, 74+1)


def extract_most_competitive_frames(sequences, prediction_dir):
    """
    Function to extract frames where the segmentation branch performs better than others, in terms of KLD.

    :param sequences: list of sequences to consider (tipically, test sequences).
    :param prediction_dir: directory where our predictions stored.
    :return: a list of tuples like (sequence, frame, gap) ordered by decreasing gap.
    """

    ret_list = []

    print 'Selecting best frames...'
    for seq in tqdm(sequences):
        kld_file = join(dreyeve_root, prediction_dir, '{:02d}'.format(seq), 'metrics', 'kld.txt')
        kld_list = read_lines_from_file(kld_file)[1:]  # remove head
        kld_list = [l[:-1].split(',') for l in kld_list]
        kld = np.array(kld_list, dtype=np.float32)

        my_kld = kld[:, [0, 8]]
        comp_kld = kld[:, [0, 6, 7]]

        # index of frames where we perform better than all competitors (by means of kld)
        good_idx = list(np.where(np.all((np.tile(my_kld[:, 1:], reps=(1, 2)) < comp_kld[:, 1:]), axis=1))[0])

        this_seq_list = [seq]*len(good_idx)
        frames_list = my_kld[good_idx, 0].tolist()
        gap = np.square(my_kld[good_idx, 1] - comp_kld[good_idx, 1]).tolist()

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
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_rgb.jpg'.format(seq, frame)), im)

        # read flow
        flow = read_image(join(dreyeve_root, 'DATA', '{:02d}'.format(seq), 'optical_flow', '{:06d}.png'.format(frame)),
                          channels_first=False,
                          color_mode='BGR',
                          dtype=np.uint8,
                          resize_dim=resize_dim)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_flow.jpg'.format(seq, frame)), flow)

        # read segmentation
        semseg = np.squeeze(np.load(join(dreyeve_root, 'DATA', '{:02d}'.format(seq),
                                         'semseg', '{:06d}.npz'.format(frame)))['arr_0'])
        semseg_rgb = cv2.resize(cv2.cvtColor(seg_to_rgb(semseg), cv2.COLOR_RGB2BGR), dsize=(960, 540))

        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_seg.jpg'.format(seq, frame)), semseg_rgb)

        # read gt and blend
        gt = read_image(join(dreyeve_root, 'DATA', '{:02d}'.format(seq), 'saliency_fix', '{:06d}.png'.format(frame+1)),
                        channels_first=False,
                        color=False,
                        dtype=np.float32,
                        resize_dim=resize_dim)

        gt = blend_map(im, gt, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_gt.jpg'.format(seq, frame)), gt)

        # read image pred and blend
        image_pred = np.squeeze(np.load(join(dreyeve_root, prediction_dir, '{:02d}'.format(seq),
                                             'image_branch', '{:06d}.npz'.format(frame)))['arr_0'])
        image_pred = cv2.resize(image_pred, dsize=resize_dim[::-1])
        image_pred = blend_map(im, image_pred, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_image_pred.jpg'.format(seq, frame)), image_pred)

        # read image pred and blend
        flow_pred = np.squeeze(np.load(join(dreyeve_root, prediction_dir, '{:02d}'.format(seq),
                                            'flow_branch', '{:06d}.npz'.format(frame)))['arr_0'])
        flow_pred = cv2.resize(flow_pred, dsize=resize_dim[::-1])
        flow_pred = blend_map(flow, flow_pred, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_flow_pred.jpg'.format(seq, frame)), flow_pred)

        # read image pred and blend
        semseg_pred = np.squeeze(np.load(join(dreyeve_root, prediction_dir, '{:02d}'.format(seq),
                                             'semseg_branch', '{:06d}.npz'.format(frame)))['arr_0'])
        semseg_pred = cv2.resize(semseg_pred, dsize=resize_dim[::-1])
        semseg_pred = blend_map(semseg_rgb, semseg_pred, factor=0.5)
        cv2.imwrite(join(this_sample_dir, 'R{:02d}_frame_{:06d}_semseg_pred.jpg'.format(seq, frame)), semseg_pred)

        stitch = [gt, im, image_pred, flow, flow_pred, semseg_rgb, semseg_pred]
        cv2.imwrite(join(stitches_dir, '{:05d}.jpg'.format(i)),
                    np.concatenate(stitch, axis=1))

# entry point
if __name__ == '__main__':

    prediction_dir = 'PREDICTIONS_2017'

    out_dir = 'competitive_figures_segmentation'

    competitive_list = extract_most_competitive_frames(dreyeve_test_seq, prediction_dir=prediction_dir)

    save_competitive_figures(out_dir=out_dir, competitive_list=competitive_list,
                             prediction_dir=prediction_dir)
