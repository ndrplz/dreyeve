import numpy as np
from pandas import read_csv
from os.path import join, exists
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC


plt.ion()


class DreyeveRun:
    """
    Single run of the DR(eye)VE dataset.
    """
    def __init__(self, dataset_data_root, num_run):
        self.num_run = num_run
        self.file_course   = join(dataset_data_root, '{:02d}'.format(self.num_run), 'speed_course_coord.txt')
        self.file_steering = join(dataset_data_root, '{:02d}'.format(self.num_run), 'steering_directions.txt')
        self.file_actions  = join(dataset_data_root, '{:02d}'.format(self.num_run), 'actions.csv')


class DreyeveDataset:
    """
    Class that models the Dreyeve dataset
    """
    def __init__(self, dataset_root):

        self.dataset_data_root = join(dataset_root, 'DATA')
        self.dataset_pred_root = join(dataset_root, 'PREDICTIONS_2017')

        self.train_runs = [DreyeveRun(self.dataset_data_root, r) for r in range(0 + 1, 38)]
        self.test_runs  = [DreyeveRun(self.dataset_data_root, r) for r in range(38, 74 + 1)]

        self.frames_each_run = 7500
        self.num_train_frames = len(self.train_runs) * self.frames_each_run
        self.num_test_frames  = len(self.test_runs)  * self.frames_each_run


def create_action_file_for_each_run(dataset):
    """
    Create action file for each run
    """
    for cur_run in dataset.test_runs:

        if not exists(cur_run.file_actions):

            speed = read_csv(cur_run.file_course, header=None, delimiter='\t').values[:, 1]     # raw speed info
            steer = read_csv(cur_run.file_steering, header=None, delimiter='\t').values[:, 0]   # STRAIGHT, RIGHT, LEFT

            assert len(speed) == len(steer)

            # Actions are derived from steering information, which is either going straight or turning. We add the
            # 'still' action, which means the car is not moving, when the speed is very low.
            where_the_car_is_still = speed < 5  # km / h
            actions = steer
            actions[where_the_car_is_still] = 'STILL'

            # Dump action file
            np.savetxt(cur_run.file_actions, actions, delimiter=',', fmt='%s')


def select_samples(dataset, how_many_samples):
    """
    Sample a certain number of examples from testing runs, uniformly distributed across actions
    
    Parameters
    ----------
    dataset: DreyeveDataset
        Instance of dreyeve dataset from which to be used for sampling data 
    how_many_samples: int
        How many examples to sample

    Returns
    -------
    overall_action_data: ndarray
        Array of `how_many_sample` rows containing example.
        
        Each example is identified by run number and `seq`,
        frame number `frame` and associated action label `label`.
    """
    # Aggregate action data for all runs in a unique structure `overall_action_data`
    overall_action_data = []
    for cur_run in dataset.test_runs:
        action_data = read_csv(cur_run.file_actions, header=None, delimiter=',').values[:, 0]
        for f, action_this_frame in enumerate(action_data):
            overall_action_data.append([cur_run.num_run, f, action_this_frame])

    # Cast to ndarray to allow slicing
    overall_action_data = np.array(overall_action_data)

    # Index each action
    idxs_still    = overall_action_data[:, 2] == 'STILL'
    idxs_straight = overall_action_data[:, 2] == 'STRAIGHT'
    idxs_left     = overall_action_data[:, 2] == 'LEFT'
    idxs_right    = overall_action_data[:, 2] == 'RIGHT'

    # Count occurrences of each action
    count_still    = np.sum(idxs_still)
    count_straight = np.sum(idxs_straight)
    count_left     = np.sum(idxs_left)
    count_right    = np.sum(idxs_right)
    assert count_left + count_right + count_still + count_straight == dataset.num_test_frames

    # Compute probabilities to sample each action uniformly
    overall_action_probs = np.zeros(shape=overall_action_data.shape[0], dtype=np.float32)
    overall_action_probs[idxs_still]    = 1 / (count_still    / dataset.num_test_frames)
    overall_action_probs[idxs_straight] = 1 / (count_straight / dataset.num_test_frames)
    overall_action_probs[idxs_left]     = 1 / (count_left     / dataset.num_test_frames)
    overall_action_probs[idxs_right]    = 1 / (count_right    / dataset.num_test_frames)
    overall_action_probs /= np.sum(overall_action_probs)

    # Sample 'how_many_samples' examples uniformly distributed across actions
    sample_idxs = np.random.choice(np.arange(0, overall_action_data.shape[0]),
                                   size=how_many_samples, p=overall_action_probs)
    return overall_action_data[sample_idxs]


if __name__ == '__main__':

    dreyeve_dataset = DreyeveDataset(dataset_root='/majinbu/public/DREYEVE/')

    # For each run in the test set
    create_action_file_for_each_run(dreyeve_dataset)

    # Sample examples uniformly distributed across actions from dreyeve testing runs
    n_samples = 10000
    samples = select_samples(dreyeve_dataset, how_many_samples=n_samples)

    activations, targets = [], []

    for seq, frame, label in samples:

        if 15 < int(frame) < 7499:  # ignore initial and final videoclip offset

            # Load network prediction
            filename = join(dreyeve_dataset.dataset_pred_root,
                            '{:02d}'.format(int(seq)),
                            'dreyeveNet', '{:06d}.npz'.format(int(frame)))
            activation = np.squeeze(np.load(filename)['arr_0'])

            # Normalize and resize to make it tractable
            activation = activation / activation.max()
            activation = cv2.resize(activation, (32, 32))

            activations.append(activation.ravel())
            targets.append(label)

    # Split sampled data into training and test
    m = len(activations)
    train_split = {'X': activations[:m // 2], 'Y': targets[:m // 2]}
    test_split  = {'X': activations[m // 2:], 'Y': targets[m//2:]}

    # Train SVM
    svm = SVC(kernel='linear')
    svm.fit(train_split['X'], train_split['Y'])

    # Predict on test set and compute metrics
    svm_pred = svm.predict(test_split['X'])
    accuracy = np.sum(svm_pred == test_split['Y']) / len(test_split['Y'])
    print('Num samples: {:06d}   ---   SVM Accuracy: {:.02f}'.format(n_samples, accuracy))
