from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


if __name__ == '__main__':

    answers_file = 'assessment_answers_header.txt'

    # Parse answers
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

    human_when_acting          = data_frame[is_human & is_acting].mean()['safeness']
    human_when_not_acting      = data_frame[is_human & not_acting].mean()['safeness']
    center_when_acting         = data_frame[is_center & is_acting].mean()['safeness']
    center_when_not_acting     = data_frame[is_center & not_acting].mean()['safeness']
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

    #####################################################################
    # Plot score distributions across different attention maps
    #####################################################################
    def plot_score_distribution_across_maps(acting_filter, title='', line_width=1, markersize=5,
                                            title_fontsize=20, label_fontsize=10, savefig=False):

        # Filter scores of the three kind of maps according to given `acting` indexes
        is_pred_scores   = sorted(np.array(data_frame[is_prediction & acting_filter]['safeness']))  # sort for plotting
        is_center_scores = sorted(np.array(data_frame[is_center     & acting_filter]['safeness']))  # sort for plotting
        is_human_scores  = sorted(np.array(data_frame[is_human      & acting_filter]['safeness']))  # sort for plotting

        # Compute probability of each score
        is_pred_scores_prob   = stats.norm.pdf(is_pred_scores,   np.mean(is_pred_scores),   np.std(is_pred_scores))
        is_center_scores_prob = stats.norm.pdf(is_center_scores, np.mean(is_center_scores), np.std(is_center_scores))
        is_human_scores_prob  = stats.norm.pdf(is_human_scores,  np.mean(is_human_scores),  np.std(is_human_scores))

        # Plot
        plt.figure(figsize=(9, 5))  # `figsize` is in (w, h) in inches
        plot_params = {'linewidth': line_width, 'markersize': markersize}
        handle_prediction, = plt.plot(is_pred_scores,   is_pred_scores_prob,   '-o', **plot_params)
        handle_center,     = plt.plot(is_center_scores, is_center_scores_prob, '-*', **plot_params)
        handle_human,      = plt.plot(is_human_scores,  is_human_scores_prob,  '-s', **plot_params)
        plt.legend([handle_prediction, handle_center, handle_human],
                   ['Model Prediction', 'Center Baseline', 'Human Groundtruth'],
                   fontsize=label_fontsize - 3)
        plt.xticks(np.arange(start=1, stop=5 + 1, step=1.0))
        plt.xlabel('Safeness score', fontsize=label_fontsize)
        plt.ylabel('Probability',    fontsize=label_fontsize)
        plt.gca().set_ylim(0, 0.4)
        plt.title(title, fontsize=title_fontsize)
        plt.tight_layout()
        if savefig:
            plt.savefig('score_distribution_{}.png'.format(title.replace(' ', '')))  # no whitespace in filename

    figure_params = {'line_width': 5, 'markersize': 12, 'title_fontsize': 24, 'label_fontsize': 18, 'savefig': False}
    plot_score_distribution_across_maps(acting_filter=True,       title='Overall',    **figure_params)
    plot_score_distribution_across_maps(acting_filter=is_acting,  title='Acting',     **figure_params)
    plot_score_distribution_across_maps(acting_filter=not_acting, title='Not Acting', **figure_params)

    #####################################################################
    # Score variance across difference drivers
    #####################################################################
    score_across_drivers = []
    for driver_id in data_frame['driver_id'].unique():
        is_cur_driver   = data_frame['driver_id'] == driver_id
        data_cur_driver = data_frame[is_human & is_cur_driver]
        score_across_drivers.append(data_cur_driver['safeness'].mean())
    print('Variance of score across different drivers: {:.02f}'.format(np.var(score_across_drivers)))

    #####################################################################
    # Average area of attention map
    #####################################################################
    attention_map_areas = {'prediction': data_frame[is_prediction]['sequence_area'].mean(),
                           'center':     data_frame[is_center]['sequence_area'].mean(),
                           'human':      data_frame[is_human]['sequence_area'].mean()}

    # Normalize dividing for max area
    max_area = np.max([v for v in attention_map_areas.values()])
    attention_map_areas = {map_type: area / max_area for (map_type, area) in attention_map_areas.items()}
    print('Average attention map area (normalized): {}'.format(attention_map_areas))
