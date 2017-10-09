"""
This script reads the file produced by `log_variance_stats.py`
and plots variances in different conditions.
"""


import numpy as np
import matplotlib.pyplot as plt
from dataset_stats.log_variance_stats import output_txt as variances_txt


columns = {
    'time_of_day': 0,
    'weather': 1,
    'scenario': 2,
    'driver': 3,
    'determinant': 4,
    'trace': 5,
    'sigma_y': 6,
    'sigma_x': 7
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

    aggregators = ['time_of_day', 'weather', 'scenario', 'driver']

    # choose an aggregator
    # aggregator = 'weather'

    for aggregator in aggregators:
        # read file
        variances = read_variances_file()

        dets = np.array(map(np.float32, variances[:, columns['determinant']]))
        traces = np.array(map(np.float32, variances[:, columns['trace']]))
        sigma_x = np.array(map(np.float32, variances[:, columns['sigma_x']]))
        sigma_y = np.array(map(np.float32, variances[:, columns['sigma_y']]))

        classes = variances[:, columns[aggregator]]

        unique_classes = np.unique(classes)

        # plot
        colors = ['#ffd700', '#21263a', '#ff2400', '#edc3c1', '#a0db8e', '#808000', '#2e4930', '#191970']

        plt.figure()
        plots = []
        for color_idx, cl in enumerate(unique_classes):
            cl_d = dets[classes == cl]
            cl_t = traces[classes == cl]
            cl_x = sigma_x[classes == cl]
            cl_y = sigma_y[classes == cl]

            print('Mean {}:\t{}\t{}'.format(cl, np.nanmean(cl_x), np.nanmean(cl_y)))

            plots.append(plt.scatter(x=cl_x, y=cl_y, marker='o', color=colors[color_idx]))

        plt.legend(plots, unique_classes, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
        plt.title('Variances by {}'.format(aggregator))
        plt.xlabel('sigma_x')
        plt.ylabel('sigma_y')
    plt.show()


# entry point
if __name__ == '__main__':
    main()
