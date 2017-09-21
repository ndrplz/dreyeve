"""
This script reads the file produced by `log_variance_stats.py`
and plots variances in different condition
"""

import numpy as np

import matplotlib.pyplot as plt

from log_variance_stats import output_txt as variances_txt

columns = {
    'time_of_day': 0,
    'weather': 1,
    'scenario': 2,
    'driver': 3,
    'determinant': 4,
    'trace': 5
}


def read_variances_file():
    """
    Reads the variances file for further plotting.
    
    Returns
    -------
    ndarray
        an array like (frame, params)
    """

    with open(variances_txt, mode='r') as f:
        variances = np.array([f.rstrip().split('\t') for f in f.readlines()])

    return variances


def main():
    """ Main function """

    # choose an aggregator
    aggregator = 'scenario'

    # read file
    variances = read_variances_file()

    dets = np.array(map(np.float32, variances[:, columns['determinant']]))
    traces = np.array(map(np.float32, variances[:, columns['trace']]))
    classes = variances[:, columns[aggregator]]

    unique_classes = np.unique(classes)

    # plot
    colors = ['#ffd700', '#21263a', '#ff2400', '#edc3c1', '#a0db8e', '#808000', '#2e4930', '#191970']

    plots = []
    for color_idx, cl in enumerate(unique_classes):
        cl_d = dets[classes == cl]
        cl_t = traces[classes == cl]

        print('Mean {}:\t{}\t{}'.format(cl, np.nanmean(cl_d), np.nanmean(cl_t)))

        plots.append(plt.scatter(x=cl_d, y=cl_t, marker='o', color=colors[color_idx]))

    plt.legend(plots, unique_classes, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
    plt.title('Variances by {}'.format(aggregator))
    plt.xlabel('Determinant of covariance matrix')
    plt.ylabel('Trace of covariance matrix')
    plt.show()


# entry point
if __name__ == '__main__':
    main()
