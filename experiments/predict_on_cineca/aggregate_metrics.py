import numpy as np

from os.path import join
from glob import glob

if __name__ == '__main__':

    predictions_dir = 'F:\DREYEVE\PREDICTIONS_2017'

    files_to_merge = ['metrics/kld_mean.txt', 'metrics/cc_mean.txt', 'ablation/kld_mean.txt', 'ablation/cc_mean.txt']

    for fm in files_to_merge:
        metrics = []
        header = ''
        for i in glob(join(predictions_dir, '**', fm)):
            with open(i) as f:
                lines = f.readlines()
            header = lines[0]
            numbers = [s.split(',') for s in lines[1:]]
            metrics.append(numbers[0])
        means = np.mean(np.array(metrics, dtype=np.float32), axis=0, keepdims=False)
        with open(join(predictions_dir, fm.replace('/', '_')), mode='w') as f:
            f.write(header)
            f.write(('{},'*means.size).format(*list(means)))

