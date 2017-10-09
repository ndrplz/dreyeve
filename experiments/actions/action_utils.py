"""
Utilities for improve code readability in `predict_actions_with_SVM.py`
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists


class DreyeveRun:
    """
    Single run of the DR(eye)VE dataset.
    """

    def __init__(self, dataset_data_root, num_run):
        self.num_run = num_run
        self.file_course = join(dataset_data_root, '{:02d}'.format(self.num_run), 'speed_course_coord.txt')
        self.file_steering = join(dataset_data_root, '{:02d}'.format(self.num_run), 'steering_directions.txt')
        self.file_actions = join(dataset_data_root, '{:02d}'.format(self.num_run), 'actions.csv')


class DreyeveDataset:
    """
    Class that models the Dreyeve dataset
    """

    def __init__(self, dataset_root):
        self.dataset_data_root = join(dataset_root, 'DATA')
        self.dataset_pred_root = join(dataset_root, 'PREDICTIONS_2017')

        self.train_runs = [DreyeveRun(self.dataset_data_root, r) for r in range(0 + 1, 38)]
        self.test_runs = [DreyeveRun(self.dataset_data_root, r) for r in range(38, 74 + 1)]

        self.frames_each_run = 7500
        self.num_train_frames = len(self.train_runs) * self.frames_each_run
        self.num_test_frames = len(self.test_runs) * self.frames_each_run


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
