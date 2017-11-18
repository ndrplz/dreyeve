"""
Some statistics utils.
"""


import numpy as np
from os.path import join


def expectation_2d(pdf):
    """
    Computes the statistical expectation of a pdf defined
    over two discrete random variables.
    
    Parameters
    ----------
    pdf: ndarray
        a numpy 2-dimensional array with probability for each (x, y).

    Returns
    -------
    ndarray
        the expectation for the x and y random variables.
    """

    h, w = pdf.shape

    pdf = np.float32(pdf)
    pdf /= np.sum(pdf)

    x_range = range(0, w)
    y_range = range(0, h)

    cols, rows = np.meshgrid(x_range, y_range)
    grid = np.stack((rows, cols), axis=-1)

    weighted_grid = pdf[..., None] * grid  # broadcasting

    E = np.apply_over_axes(np.sum, weighted_grid, axes=[0, 1])
    E = np.squeeze(E)

    return E


def covariance_matrix_2d(pdf):
    """
    Computes the covariance matrix of a 2-dimensional gaussian
    fitted over a joint pdf of two discrete random variables.

    Parameters
    ----------
    pdf: ndarray
        a numpy 2-dimensional array with probability for each (x, y).

    Returns
    -------
    ndarray
        the covariance matrix.
    """

    h, w = pdf.shape

    pdf = np.float32(pdf)
    pdf /= np.sum(pdf)

    x_range = range(0, w)
    y_range = range(0, h)

    cols, rows = np.meshgrid(x_range, y_range)
    grid = np.stack((rows, cols), axis=-1)

    mu = expectation_2d(pdf)

    grid = np.float32(grid)

    # remove mean
    grid -= mu[None, None, :]
    grid_flat = np.reshape(grid, newshape=(-1, 2))

    # in computing the dot product, pdf has to be counted one (outside the square!)
    cov = np.dot(grid_flat.T, grid_flat * np.reshape(pdf, -1)[..., None])
    return cov


def read_dreyeve_design(dreyeve_root):
    """
    Reads the whole dr(eye)ve design.

    Returns
    -------
    ndarray
        the dr(eye)ve design in the form (sequences, params).
    """

    with open(join(dreyeve_root, 'dr(eye)ve_design.txt')) as f:
        dreyeve_design = np.array([l.rstrip().split('\t') for l in f.readlines()])

    return dreyeve_design
