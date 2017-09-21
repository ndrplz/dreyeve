"""
Computes the variance of the fixation map (as a distribution) 
for each groundtruth frame.
"""

import numpy as np

from os.path import join
from glob import glob

import skimage.io as io

from tqdm import tqdm

from stats_utils import covariance_matrix_2d

dreyeve_root = '/majinbu/public/DREYEVE'
output_txt = 'variances.txt'


def read_dreyeve_design():
    """
    Reads the whole dr(eye)ve design.
    
    Returns
    -------
    ndarray
        the dr(eye)ve design in the form (sequences, params).
    """

    with open(join(dreyeve_root, 'dr(eye)ve_design.txt')) as f:
        dreyeve_design = np.array([l.rstrip().split('\t') for l in f.readlines()])

    return dreyeve_design


def write_line_on_file(line, f):
    """
    Writes a line on an opened file.
    
    Parameters
    ----------
    line: Sized
        a list of string to write separated by '\t'.
    f: BinaryIO
        an already opened file.
    """

    f.write(('{}\t' * len(line)).rstrip().format(*line))
    f.write('\n')


def main():
    """ Main function """
    dreyeve_design = read_dreyeve_design()

    with open(output_txt, mode='w') as f:
        for line, seq in enumerate(dreyeve_design[:, 0]):

            seq_dir = join(dreyeve_root, 'DATA', seq, 'saliency_fix')

            frames = sorted(glob(join(seq_dir, '*.png')))
            n_frames = len(frames)
            for frame_idx in tqdm(range(0, n_frames, 75), desc='Sequence {}'.format(seq)):

                frame = frames[frame_idx]
                fixation_map = io.imread(frame)

                cov = covariance_matrix_2d(fixation_map)

                write_line_on_file([dreyeve_design[line, 1],
                                    dreyeve_design[line, 2],
                                    dreyeve_design[line, 3],
                                    dreyeve_design[line, 4],
                                    np.linalg.det(cov),
                                    np.trace(cov)], f)


# entry point
if __name__ == '__main__':
    main()
