"""
This script proposes some test sequence of semantic segmentation to the attention
of a human inspector, that has to decide if the segmentation is good or not.
"""

import numpy as np
import cv2

from os.path import join
from glob import glob

from random import choice

from train.utils import seg_to_colormap
from train.config import dreyeve_test_seq


if __name__ == '__main__':

    dreyeve_dir = 'Z:/DATA'
    sequences = dreyeve_test_seq

    good_sequences = []
    for seq in sequences:

        print 'Evaluating sequence {}'.format(seq)

        sequence_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'semseg')
        seg_list = glob(join(sequence_dir, '*.npz'))

        key = 0
        while key != ord('b') and key != ord('g'):

            semseg = np.argmax(np.squeeze(np.load(choice(seg_list))['arr_0']), axis=0)
            semseg_img = seg_to_colormap(semseg, channels_first=False)
            semseg_img = cv2.resize(semseg_img, dsize=None, fx=2, fy=2)
            semseg_img = cv2.cvtColor(semseg_img, cv2.COLOR_RGB2BGR)

            cv2.imshow('SEMSEG', semseg_img)
            key = cv2.waitKey(3000)

        if key == ord('g'):
            good_sequences.append(seq)
            print good_sequences

    print good_sequences
