import numpy as np


def kld_numeric(y_true, y_pred):
    """
    Function to evaluate Kullback-Leiber divergence (sec 4.2.3 of [1]) on two samples.
    The two distributions are numpy arrays having arbitrary but coherent shapes.

    :param y_true: groundtruth.
    :param y_pred: predictions.
    :return: numeric kld
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    eps = np.finfo(np.float64).eps

    P = y_pred / (eps + np.sum(y_pred))  # prob
    Q = y_true / (eps + np.sum(y_true))  # prob

    kld = np.sum(Q * np.log(eps + Q / (eps + P)))

    return kld


def cc_numeric(y_true, y_pred):
    """
    Function to evaluate Pearson's correlation coefficient (sec 4.2.2 of [1]) on two samples.
    The two distributions are numpy arrays having arbitrary but coherent shapes.

    :param y_true: groundtruth.
    :param y_pred: predictions.
    :return: numeric cc.
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    eps = np.finfo(np.float64).eps

    P = y_pred / (eps + np.sum(y_pred))  # normalization
    Q = y_true / (eps + np.sum(y_true))  # normalization

    N = P.size
    cc = np.dot(P.flatten() - np.mean(P), Q.flatten() - np.mean(Q)) / (eps + np.std(P) * np.std(Q)) / N

    return cc


def ig_numeric(y_true, y_pred, y_base):
    """
    Function to evaluate the information gain (sec 4.1.3 of [1]).
    The two distributions are numpy arrays having arbitrary but coherent shapes.

    :param y_true: groundtruth.
    :param y_pred: predictions.
    :param y_base: baseline.
    :return: numeric ig.
    """

    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    y_base = y_base.astype(np.float32)

    eps = np.finfo(np.float32).eps

    P = y_pred / (eps + np.sum(y_pred))  # prob
    Q = y_true / (eps + np.sum(y_true))  # prob
    B = y_base / (eps + np.sum(y_base))  # prob

    N = P.size
    ig = np.sum(Q * (np.log2(eps + P) - np.log2(eps + B))) / N

    return ig

"""
REFERENCES:
[1] @article{salMetrics_Bylinskii,
  title     = {What do different evaluation metrics tell us about saliency models?},
  author    = {Zoya Bylinskii and Tilke Judd and Aude Oliva and Antonio Torralba and Fr{\'e}do Durand},
  journal   = {arXiv preprint arXiv:1604.03605},
  year      = {2016}
}
"""
