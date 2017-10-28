from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd


if __name__ == '__main__':

    answers_file = 'assessment_answers.txt'

    # todo add the following header to the logs recorded in 'show_attentional_video.py':
    # todo subject_id,video_filename,driver_id,which_map,seq,start_frame,end_frame,is_acting,sequence_area,count_acting,safeness,turing

    data_frame = pd.read_csv(answers_file)

    #####################################################################
    # Prepare indexes
    #####################################################################

    is_human      = data_frame['which_map'] == 'groundtruth'
    is_center     = data_frame['which_map'] == 'central_baseline'
    is_prediction = data_frame['which_map'] == 'prediction'
    is_machine    = is_center | is_prediction

    subject_thinks_human   = data_frame['turing'] == 'Human'
    subject_thinks_machine = data_frame['turing'] == 'AI'

    is_acting     = data_frame['is_acting'] == 'acting'
    not_acting    = data_frame['is_acting'] == 'non_acting'

    #####################################################################
    # Overall Statistics
    #####################################################################

    num_rows = len(data_frame)
    num_subjects = len(data_frame['subject_id'].unique())

    overall_safeness_mean  = data_frame['safeness'].mean()
    overall_safeness_var   = data_frame['safeness'].var()

    is_machine_safeness_mean  = data_frame[is_machine].mean()['safeness']
    is_machine_safeness_var   = data_frame[is_machine].var()['safeness']
    is_human_safeness_mean    = data_frame[is_human].mean()['safeness']
    is_human_safeness_var     = data_frame[is_human].var()['safeness']

    is_center_safeness_mean     = data_frame[is_center].mean()['safeness']
    is_center_safeness_var      = data_frame[is_center].var()['safeness']
    is_prediction_safeness_mean = data_frame[is_prediction].mean()['safeness']
    is_prediction_safeness_var  = data_frame[is_prediction].var()['safeness']

    subject_thinks_machine_safeness_mean = data_frame[subject_thinks_machine].mean()['safeness']
    subject_thinks_machine_safeness_var  = data_frame[subject_thinks_machine].var()['safeness']
    subject_thinks_human_safeness_mean   = data_frame[subject_thinks_human].mean()['safeness']
    subject_thinks_human_safeness_var    = data_frame[subject_thinks_human].var()['safeness']

    center_when_acting     = data_frame[is_center & is_acting].mean()['safeness']
    center_when_not_acting = data_frame[is_center & not_acting].mean()['safeness']
    prediction_when_acting     = data_frame[is_prediction & is_acting].mean()['safeness']
    prediction_when_not_acting = data_frame[is_prediction & not_acting].mean()['safeness']

    #####################################################################
    # Turing Test - Confusion Matrix
    #####################################################################

    TP_idx = subject_thinks_human   & is_human
    TN_idx = subject_thinks_machine & is_machine
    FN_idx = subject_thinks_machine & is_human
    FP_idx = subject_thinks_human   & is_machine

    TP, TN, FN, FP = [np.sum(x) for x in [TP_idx, TN_idx, FN_idx, FP_idx]]
    confusion_matrix = np.array([[TP, FN], [FP, TN]])

    guessed_right = (TP + TN) / num_rows
    print('Turing test result: guessed right in {:2d}% of cases.'.format(int(guessed_right * 100)))

    #####################################################################
    # Turing Test - Score distribution across confusion matrix
    #####################################################################

    safeness_range = [1, 2, 3, 4, 5]
    safeness_distribution = np.zeros(shape=(4, len(safeness_range)))

    for col, safeness_score in enumerate(safeness_range):
        safeness_distribution[0, col] = np.sum(data_frame[TP_idx]['safeness'] == safeness_score)
        safeness_distribution[1, col] = np.sum(data_frame[TN_idx]['safeness'] == safeness_score)
        safeness_distribution[2, col] = np.sum(data_frame[FP_idx]['safeness'] == safeness_score)
        safeness_distribution[3, col] = np.sum(data_frame[FN_idx]['safeness'] == safeness_score)
        safeness_distribution[:, col] /= np.sum(safeness_distribution[:, col])
    safeness_distribution_frame = pd.DataFrame(safeness_distribution, index=['TP', 'TN', 'FP', 'FN'])
    print(safeness_distribution_frame)
