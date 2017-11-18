"""
This script mines all frames where the segmentation branch works better than others,
in order to find a pattern.
"""

import numpy as np

from os.path import join, basename, dirname
from glob import glob

from tqdm import tqdm

from train.utils import read_lines_from_file
from dataset_stats.stats_utils import read_dreyeve_design


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

        my_kld = kld[:, [0, 8]]  # kld of segmentation branch
        comp_kld = kld[:, [0, 6, 7]]  # kld of other branches

        # index of frames where we perform better than all competitors (by means of kld)
        good_idx = list(np.where(np.all((np.tile(my_kld[:, 1:], reps=(1, 2)) < comp_kld[:, 1:]), axis=1))[0])

        this_seq_list = [seq]*len(good_idx)
        frames_list = my_kld[good_idx, 0].astype(np.int32).tolist()
        gap = np.square(my_kld[good_idx, 1] - comp_kld[good_idx, 1]).tolist()

        ret_list += zip(this_seq_list, frames_list, gap)

    return ret_list


def read_actions(dreyeve_root):

    # dictionary like {seq: np.array of 7500 steering dirs}
    ret_dict = {}

    for actions_file in glob(join(dreyeve_root, 'DATA', '**', 'actions.csv')):

        # sequence number
        seq = int(basename(dirname(actions_file)))

        with open(actions_file) as f:
            actions = np.array([l.rstrip() for l in f.readlines()])

        ret_dict[seq] = actions

    return ret_dict


# entry point
if __name__ == '__main__':

    prediction_dir = 'PREDICTIONS_2017'

    # read competitive list = [(seq, frame, gap with others)]
    competitive_list = extract_most_competitive_frames(dreyeve_test_seq, prediction_dir=prediction_dir)

    # read dreyeve design
    dreyeve_design = read_dreyeve_design(dreyeve_root)
    dreyeve_design = {int(row[0]): row[1:] for row in dreyeve_design}

    # read steering directions
    actions = read_actions(dreyeve_root)

    # compute priors over actions
    all_actions = np.concatenate([actions[seq] for seq in actions], axis=0)
    actions_priors = {'STRAIGHT': float(sum(all_actions == 'STRAIGHT')) / all_actions.shape[0],
                      'STILL': float(sum(all_actions == 'STILL')) / all_actions.shape[0],
                      'RIGHT': float(sum(all_actions == 'RIGHT')) / all_actions.shape[0],
                      'LEFT': float(sum(all_actions == 'LEFT')) / all_actions.shape[0]}

    # initialize histograms
    time_of_day_histo = {'Morning': 0, 'Evening': 0, 'Night': 0}
    weather_histo = {'Sunny': 0, 'Cloudy': 0, 'Rainy': 0}
    landscape_histo = {'Countryside': 0, 'Highway': 0, 'Downtown': 0}
    actions_histo = {'LEFT': 0, 'STRAIGHT': 0, 'STILL': 0, 'RIGHT': 0, }


    # loop over lucky frames
    for seq, frame, _ in competitive_list:

        # update histograms
        time_of_day_histo[dreyeve_design[seq][0]] += 1
        weather_histo[dreyeve_design[seq][1]] += 1
        landscape_histo[dreyeve_design[seq][2]] += 1

        actions_histo[actions[seq][frame]] += 1

    # print results
    n_lucky_frames = float(len(competitive_list))
    print('TIME OF DAY:\n{}'.format({seq: time_of_day_histo[seq] / n_lucky_frames for seq in time_of_day_histo}))
    print('WEATHER:\n{}'.format({seq: weather_histo[seq] / n_lucky_frames for seq in weather_histo}))
    print('LANDSCAPE:\n{}'.format({seq: landscape_histo[seq] / n_lucky_frames for seq in landscape_histo}))
    print('=======================================================================================')
    print('ACTIONS:\n{}'.format({seq: actions_histo[seq] / n_lucky_frames for seq in actions_histo}))
    print('ACTIONS_PRIORS:\n{}'.format(actions_priors))
