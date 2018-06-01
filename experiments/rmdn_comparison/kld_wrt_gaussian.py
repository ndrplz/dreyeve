"""
This file measures the KLD between RMDN predictions with respect to central gaussian baseline.
"""
from os.path import join

import numpy as np
from computer_vision_utils.io_helper import read_image
from metrics.metrics import kld_numeric
from tqdm import tqdm


def main():

    prediction_root = '/majinbu/public/DREYEVE/PREDICTIONS_RMDN'
    gaussian_path = '/majinbu/public/DREYEVE/DATA/dreyeve_mean_train_gt_fix.png'

    central_gaussian = read_image(gaussian_path, channels_first=False, color=False)

    test_sequences = range(38, 74+1)
    klds = []
    for seq in test_sequences:

        seq_root = join(prediction_root, '{:02d}'.format(seq), 'output')

        for frame_idx in tqdm(range(15, 7500, 5), desc='Processing sequence {:02d}...'.format(seq)):
            img = read_image(join(seq_root, '{:06d}.png'.format(frame_idx)), channels_first=False, color=False,
                             resize_dim=(1080, 1920))

            klds.append(kld_numeric(img, central_gaussian))

    print np.mean(klds)


if __name__ == '__main__':
    main()
