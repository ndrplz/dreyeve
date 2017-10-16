"""
This script is used to construct videos mimicking attentional behavior of
humans, deep model, central baseline. Such videos will be then used for
the quality assessment (Sec 5.4 of the paper).
"""

import numpy as np
from os.path import join


# parameters
dreyeve_root = '/majinbu/public/DREYEVE'
subsequences_txt = join(dreyeve_root, 'subsequences.txt')
n_frames = 1000
shape = (1080 // 2, 1920 // 2)


def get_driver_for_sequence(seq):
    """
    This function returns the driver id of a given sequence.

    Parameters
    ----------
    seq: int
        the sequence number
    Returns
    -------
    str
        the driver id
    """

    with open(join(dreyeve_root, 'dr(eye)ve_design.txt')) as f:
        dreyeve_design = np.array([f.rstrip().split('\t') for f in f.readlines()])

    row = np.where(dreyeve_design[:, 0] == '{:02d}'.format(int(seq)))[0][0]
    driver_id = dreyeve_design[row, 4]

    return driver_id


def get_random_clip():
    """
    This function returns a random clip.
    
    Returns
    -------
    tuple
        a tuple like (seq, start_frame, contains_acting).
    """

    with open(subsequences_txt, mode='r') as f:
        subsequences = np.array([l.rstrip().split('\t') for l in f.readlines()])

    acting_subseqs = subsequences[subsequences[:, 3] == 'k', :3]
    acting_subseqs = np.int32(acting_subseqs)

    sequences = range(38, 74 + 1)
    seq_probs = np.array([np.shape(acting_subseqs[acting_subseqs[:, 0] == s])[0] for s in sequences], dtype=np.float32)
    seq_probs /= np.sum(seq_probs)

    contains_acting = np.random.choice(['acting', 'non_acting'])

    while True:
        if contains_acting == 'acting':
            seq = np.random.choice(sequences, p=seq_probs)

            acting_subseqs = acting_subseqs[acting_subseqs[:, 0] == seq]

            start_probs = np.zeros(shape=7500, dtype=np.float32)
            for _, start, stop in acting_subseqs:
                start = max(0, start - n_frames)
                stop = max(0, stop)

                start_probs[start:stop] += 1

            start_probs[-n_frames:] = 0
            start_probs[0] = 0
            start_probs /= np.sum(start_probs)

            start = np.random.choice(range(0, 7500), p=start_probs)

        else:
            seq = np.random.choice(sequences)

            acting_subseqs = acting_subseqs[acting_subseqs[:, 0] == seq]

            start_probs = np.ones(shape=7500, dtype=np.float32)
            for _, start, stop in acting_subseqs:
                start = max(0, start - n_frames)

                start_probs[start:stop] = 0

            start_probs[-n_frames:] = 0
            start_probs[0] = 0
            start_probs /= np.sum(start_probs)

            start = np.random.choice(range(0, 7500), p=start_probs)

        if start != 0:  # exit
            break

    # count acting frames
    is_frame_acting = np.zeros(shape=(7500,), dtype=np.int32)
    for _, acting_start, acting_stop in acting_subseqs:
        is_frame_acting[acting_start:acting_stop] = 1

    count_acting = sum(is_frame_acting[start:start + n_frames])

    return seq, start, contains_acting, count_acting
