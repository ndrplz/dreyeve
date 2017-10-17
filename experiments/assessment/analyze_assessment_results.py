from __future__ import division
from __future__ import print_function
import numpy as np
from pandas import read_csv


if __name__ == '__main__':

    answers_file = 'assessment_answers.txt'

    # todo add this header when writing logs in 'show_attentional_video.py'
    # todo subject_id, video_filename, driver_id, which_map, seq, start_frame, end_frame, is_acting, sequence_area, count_acting, safeness, turing

    data_frame = read_csv(answers_file)

    #####################################################################
    # Overall Statistics
    #####################################################################
    num_rows = len(data_frame)
    num_subjects = len(data_frame['subject_id'].unique())

    overall_safeness_mean  = data_frame['safeness'].mean()
    overall_safeness_var   = data_frame['safeness'].var()
    ai_safeness_mean       = data_frame[data_frame['turing'] == 'AI'].mean()['safeness']
    ai_safeness_var        = data_frame[data_frame['turing'] == 'AI'].var()['safeness']
    human_safeness_mean    = data_frame[data_frame['turing'] == 'Human'].mean()['safeness']
    human_safeness_var     = data_frame[data_frame['turing'] == 'Human'].var()['safeness']

    #####################################################################
    # Turing Test - Confusion Matrix (Human is positive, AI is negative)
    #####################################################################

    is_human   = data_frame['which_map'] == 'groundtruth'
    is_machine = (data_frame['which_map'] == 'prediction') | (data_frame['which_map'] == 'central_baseline')
    subject_thinks_human   = data_frame['turing'] == 'Human'
    subject_thinks_machine = data_frame['turing'] == 'AI'

    TP_idx = subject_thinks_human   & is_human
    TN_idx = subject_thinks_machine & is_machine
    FN_idx = subject_thinks_machine & is_human
    FP_idx = subject_thinks_human   & is_machine

    TP, TN, FN, FP = [np.sum(x) for x in [TP_idx, TN_idx, FN_idx, FP_idx]]
    confusion_matrix = np.array([[TP, FN], [FP, TN]])

    #####################################################################
    # Turing Test - Score distribution across confusion matrix
    #####################################################################
    # `safeness_distribution` table looks like this
    # +----+--------+--------+--------+---------+--------+
    # |    | Safe=1 | Safe=2 | Safe=3 | Safe=4  | Safe=5 |
    # +----+--------+--------+--------+---------+--------+
    # | TP |        |        |        |         |        |
    # | TN |        |        |        |         |        |
    # | FP |        |        |        |         |        |
    # | FN |        |        |        |         |        |
    # +----+--------+--------+--------+---------+--------+
    #####################################################################
    safeness_range = [1, 2, 3, 4, 5]
    safeness_distribution = np.zeros(shape=(4, len(safeness_range)))

    for col, safeness_score in enumerate(safeness_range):
        safeness_distribution[0, col] = np.sum(data_frame[TP_idx]['safeness'] == safeness_score)
        safeness_distribution[1, col] = np.sum(data_frame[TN_idx]['safeness'] == safeness_score)
        safeness_distribution[2, col] = np.sum(data_frame[FP_idx]['safeness'] == safeness_score)
        safeness_distribution[3, col] = np.sum(data_frame[FN_idx]['safeness'] == safeness_score)
        safeness_distribution[:, col] /= np.sum(safeness_distribution[:, col])
    print(safeness_distribution)
