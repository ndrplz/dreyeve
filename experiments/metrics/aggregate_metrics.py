import numpy as np
from os.path import join


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':

    # do not change these
    sequences_train = xrange(1, 38)
    sequences_test = xrange(38, 74 + 1)
    predictions_dir = '/majinbu/public/DREYEVE/PREDICTIONS_CENTRAL_CROP'

    # change these
    metrics_to_merge = ['metrics/kld_mean.txt', 'metrics/cc_mean.txt', 'metrics/ig_mean.txt']
    mode = 'test'

    assert mode in ['train', 'test', 'only_good_semseg'], 'Non valid mode: {}'.format(mode)

    sequences = []
    if mode == 'train':
        sequences = sequences_train
    elif mode == 'test':
        sequences = sequences_test
    elif mode == 'only_good_semseg':
        sequences = [40, 44, 47, 49, 50, 60, 63, 64, 69, 70]

    for metric_filename in metrics_to_merge:
        metrics = []
        header = ''
        frames = 0
        # read metric for all sequences
        for metric_file in [join(predictions_dir, '{:02d}'.format(seq), metric_filename) for seq in sequences]:
            # count how many frames we have in this sequence
            # pardon, this part was added after
            cur_frames = file_len(metric_file.replace('_mean', '')) - 1

            # read
            with open(metric_file) as f:
                lines = f.readlines()
            # read header
            header = lines[0]
            # read numbers
            numbers = [s.split(',') for s in lines[1:]][0]
            numbers = [float(n)*cur_frames for n in numbers if n != '']

            metrics.append(numbers)
            frames += cur_frames

        # mean on all sequences
        means = np.sum(np.array(metrics, dtype=np.float32), axis=0, keepdims=False)
        means /= frames
        # write
        with open(join(predictions_dir, '{}_{}'.format(mode, metric_filename.replace('/', '_'))),
                  mode='w') as f:
            f.write(header)
            f.write(('{},'*means.size).format(*list(means)))
