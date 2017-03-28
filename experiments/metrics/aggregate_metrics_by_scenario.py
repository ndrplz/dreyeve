"""
This script is meant to aggregate metrics in different triplets of scenarios, in order to make
one plot for each scenario showing the kld distance in different subcontexts like (day/rainy, night/sunny etc.)
"""
import numpy as np

from os.path import join
from train.config import dreyeve_test_seq

import operator

if __name__ == '__main__':

    dreyeve_dir = 'Z:/'
    scenarios = ['Downtown', 'Countryside', 'Highway']
    metric_to_merge = 'metrics/kld_mean.txt'

    # read design file and create dictionaries
    with open(join(dreyeve_dir, 'dr(eye)ve_design.txt')) as f:
        content = f.readlines()

    design = {'Downtown': {}, 'Countryside': {}, 'Highway': {}}
    for c in content:
        seq, time_of_day, wheather, scenario, driver, set, _ = c.split()
        seq = int(seq)

        if seq in dreyeve_test_seq:
            d = design[scenario]

            key = (time_of_day, wheather)
            val = d.get(key)
            if val:
                d[key].append(seq)
            else:
                d[key] = [seq]

    for scenario in design:
        scenario_dict = design[scenario]

        for key, sequences in scenario_dict.iteritems():
            metrics = []
            for seq in sequences:
                header = ''
                # read metric for all sequences
                metric_file = join(dreyeve_dir, 'PREDICTIONS_2017', '{:02d}'.format(seq),
                                   metric_to_merge)

                # read
                with open(metric_file) as f:
                    lines = f.readlines()
                # read header
                header = lines[0]
                # read numbers
                numbers = [s.split(',') for s in lines[1:]][0]
                numbers = [n for n in numbers if n != '']
                metrics.append(numbers)
            # mean on all sequences of this context
            means = np.mean(np.array(metrics, dtype=np.float32), axis=0, keepdims=False)
            scenario_dict[key] = means[0]

    for key, d in design.iteritems():
        print key

        # sort by kld
        sorted_d = sorted(d.items(), key=operator.itemgetter(1))
        print sorted_d
