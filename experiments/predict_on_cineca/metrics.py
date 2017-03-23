import numpy as np
import cv2


def kld_numeric(y_true, y_pred):
    """
    Function to evaluate Kullback-Leiber divergence (sec 4.2.3 of [1]) on two samples.
    The two distributions are numpy arrays having arbitrary but coherent shapes.

    :param y_true: groundtruth.
    :param y_pred: predictions.
    :return: numeric kld
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    eps = np.finfo(np.float32).eps

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
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)

    eps = np.finfo(np.float32).eps

    cv2.normalize(y_pred, dst=y_pred, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(y_true, dst=y_true, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    y_pred = y_pred.ravel()
    y_true = y_true.ravel()

    y_pred = (y_pred - np.mean(y_pred)) / (eps + np.std(y_pred))
    y_true = (y_true - np.mean(y_true)) / (eps + np.std(y_true))

    cc = np.corrcoef(y_pred, y_true)

    return cc[0][1]


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
    ig = np.sum(Q * (np.log2((eps + P)/(eps + B)))) / N

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
