import numpy as np

from stats_utils import read_dreyeve_design
import matplotlib.pyplot as plt


def count_subsequences_each_sequence(subsequences_file, sequences):
    """
    Returns
    -------
    dict
        a dictionary like:
            {dreyeve_seq: np.array([count_acting, count_errors, ...])}
    """

    ret = {s: np.zeros(5, dtype=np.int32) for s in sequences}  # normal, acting, errors, inattentive, interesting

    # read subsequences file and populate sequence histogram
    with open(subsequences_file, mode='r') as f:
        for line in f.readlines():
            seq, start, end, kind = line.rstrip().split('\t')
            seq, start, end = int(seq), int(start), int(end)

            if kind == 'k':  # acting
                ret[seq][1] += end - start + 1
            elif kind == 'e':  # error
                ret[seq][2] += end - start + 1
            elif kind == 'i':  # inattentive
                ret[seq][3] += end - start + 1
            elif kind == 'u':  # uninteresting
                ret[seq][4] += end - start + 1
            else:
                raise ValueError

    # count `normal` central frames
    for key, value in ret.iteritems():
        value[0] = 7500 - np.sum(value[1:])

    return ret


def main():
    """ Main function """

    dreyeve_root = '/majinbu/public/DREYEVE'

    subsequences_file = '/majinbu/public/DREYEVE/subsequences.txt'
    sequences = np.arange(1, 74+1)

    subsequences_count = count_subsequences_each_sequence(subsequences_file, sequences)

    dreyeve_design = read_dreyeve_design(dreyeve_root=dreyeve_root)
    dreyeve_design = {int(s[0]): s[1:] for s in dreyeve_design}

    # aggregate sequences by weather
    weathers = ['Sunny', 'Cloudy', 'Rainy']
    aggr_col = 1
    weather_hist = {w: np.zeros(5, dtype=np.int32) for w in weathers}

    for s in sequences:
        this_sequence_weather = dreyeve_design[s][aggr_col]

        weather_hist[this_sequence_weather] += subsequences_count[s]

    for w in weathers:
        print('{}: {}'.format(w, weather_hist[w].astype(np.float32) / np.sum(weather_hist[w])))


if __name__ == '__main__':
    main()


