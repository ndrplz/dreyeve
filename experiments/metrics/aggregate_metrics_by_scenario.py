"""
This script is meant to aggregate metrics in different triplets of scenarios, in order to make
one plot for each scenario showing the kld distance in different subcontexts like (day/rainy, night/sunny etc.)
"""
from __future__ import print_function
from __future__ import division
import numpy as np
from os.path import join
from train.config import dreyeve_test_seq


def dump_dict_to_csv(dict):
    with open('dump.csv', mode='w') as f:
        for scenario, subdict in dict.iteritems():
            for subkey, value in subdict.iteritems():
                f.write('{},{},{},{}\n'.format(scenario, subkey[0], subkey[1], value))


if __name__ == '__main__':

    dreyeve_root = '/majinbu/public/DREYEVE'
    aggregate_by = 1  # 0 for time_of_day, 1 for weather, 2 for scenario
    metric_to_merge = 'metrics/kld_mean.txt'

    # read design file and create dictionaries
    with open(join(dreyeve_root, 'dr(eye)ve_design.txt')) as f:
        content = f.readlines()

    design = {}
    for c in content:
        seq, time_of_day, weather, key, driver, set, _ = c.split()
        seq = int(seq)

        possible_aggregations = [time_of_day, weather, key]

        aggr_key = possible_aggregations[aggregate_by]

        possible_aggregations.pop(aggregate_by)
        sub_key = tuple(possible_aggregations)

        if seq in dreyeve_test_seq:
            subdict = design.get(aggr_key)

            if not subdict:
                design[aggr_key] = {}
                subdict = design.get(aggr_key)

            val = subdict.get(sub_key)
            if val:
                subdict[sub_key].append(seq)
            else:
                subdict[sub_key] = [seq]

    for key in design:
        scenario_dict = design[key]

        for aggr_key, sequences in scenario_dict.iteritems():
            metrics = []
            for seq in sequences:
                header = ''
                # read metric for all sequences
                metric_file = join(dreyeve_root, 'PREDICTIONS_2017', '{:02d}'.format(seq),
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
            scenario_dict[aggr_key] = means[4:]

    for aggr_key, subdict in design.iteritems():
        print(aggr_key)

        print(np.mean(np.array(subdict.values()), axis=0))
