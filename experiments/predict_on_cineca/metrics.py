import numpy as np


def kld_numeric(y_true, y_pred):
    """
    Function to evaluate Kullback-Leiber divergence (sec 4.2.3 of [1]) on two samples.

    :param y_true: groundtruth, having shape (1, 1, h, w)
    :param y_pred: predictions, having shape (1, 1, h, w)
    :return: numeric kld
    """
    eps = np.finfo(float).eps

    P = y_pred / (eps + np.sum(y_pred))  # prob
    Q = y_true / (eps + np.sum(y_true))  # prob

    kld = np.sum(Q * np.log(eps + Q / (eps + P)))

    return kld


def cc_numeric(y_true, y_pred):
    """
    Function to evaluate Pearson's correlation coefficient (sec 4.2.2 of [1]) on two samples.

    :param y_true: groundtruth, having shape (1, 1, h, w)
    :param y_pred: predictions, having shape (1, 1, h, w)
    :return: numeric cc
    """
    eps = np.finfo(float).eps

    P = y_pred / (eps + np.sum(y_pred))  # prob
    Q = y_true / (eps + np.sum(y_true))  # prob

    n = P.size
    cc = np.dot(P.flatten() - np.mean(P), Q.flatten() - np.mean(Q)) / (eps + np.std(P) * np.std(Q)) / n

    return cc

"""
REFERENCES:
[1] @article{salMetrics_Bylinskii,
  title     = {What do different evaluation metrics tell us about saliency models?},
  author    = {Zoya Bylinskii and Tilke Judd and Aude Oliva and Antonio Torralba and Fr{\'e}do Durand},
  journal   = {arXiv preprint arXiv:1604.03605},
  year      = {2016}
}
"""
