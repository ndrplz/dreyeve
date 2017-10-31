"""
This script merges all the new groundtruth fixation maps and saves in an image.
Such mean map will be used as baseline in experiments.
ONLY TRAINING SEQUENCES ARE CONSIDERED.
"""
from __future__ import print_function
import cv2
import numpy as np
from os.path import join
from glob import glob
from tqdm import tqdm
from train.config import dreyeve_train_seq


if __name__ == '__main__':

    dreyeve_data_root = '/majinbu/public/DREYEVE/DATA'
    sequences = dreyeve_train_seq

    mean_img = np.zeros(shape=(1080, 1920), dtype=np.float32)
    n_images = 0
    for seq in sequences:

        print('Processing sequence {}'.format(seq))

        sequence_dir = join(dreyeve_data_root, '{:02d}'.format(seq), 'saliency_fix')
        gt_list = glob(join(sequence_dir, '*.png'))

        for gt_img in tqdm(gt_list):
            mean_img += cv2.imread(gt_img, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        n_images += len(gt_list)

        cv2.imshow('current_mean', ((mean_img / n_images) * 255).astype(np.uint8))
        cv2.waitKey(1)

    mean_img /= n_images
    cv2.imwrite(join(dreyeve_data_root, 'dreyeve_mean_train_gt_fix.png'), (mean_img * 255).astype(np.uint8))
