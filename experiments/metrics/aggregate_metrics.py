import numpy as np

import os
from os.path import join


if __name__ == '__main__':

    # do not change these
    sequences_train = xrange(1, 38 + 1)
    sequences_test = xrange(39, 74 + 1)
    predictions_dir = 'Z:/PREDICTIONS_2017'

    # change these
    metrics_to_merge = ['metrics/kld_mean.txt', 'metrics/cc_mean.txt', 'metrics/ig_mean.txt',
                        'ablation/kld_mean.txt', 'ablation/cc_mean.txt', 'ablation/ig_mean.txt']
    mode = 'test'

    assert mode in ['train', 'test'], 'Non valid mode: {}'.format(mode)

    sequences = sequences_train if mode == 'train' else sequences_test

    for metric_filename in metrics_to_merge:
        metrics = []
        header = ''
        # read metric for all sequences
        for metric_file in [join(predictions_dir, '{:02d}'.format(seq), metric_filename) for seq in sequences]:
            # read
            with open(metric_file) as f:
                lines = f.readlines()
            # read header
            header = lines[0]
            # read numbers
            numbers = [s.split(',') for s in lines[1:]][0]
            numbers = [n for n in numbers if n != '']
            metrics.append(numbers)
        # mean on all sequences
        means = np.mean(np.array(metrics, dtype=np.float32), axis=0, keepdims=False)
        # write
        with open(join(predictions_dir, '{}_{}'.format(mode, metric_filename.replace('/', '_'))),
                  mode='w') as f:
            f.write(header)
            f.write(('{},'*means.size).format(*list(means)))

