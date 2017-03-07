import keras.backend as K


def saliency_loss(name, mse_beta=None):
    """
    Returns loss for the saliency task.

    :param name: string identifier of loss function.
    :param mse_beta: regularizer for weighted mse.
    :return: the loss symbolic function.
    """
    assert name in ['mse', 'sse', 'nss', 'simo', 'kld'], 'Unknown loss function: {}'.format(name)

    # K.mean: axis can be None - in which case the mean is computed along all axes(like numpy)
    # see http://deeplearning.net/software/theano/library/tensor/basic.html
    def mean_squared_error(y_true, y_pred):
        """
        Mean squared error loss.

        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss symbolic value.
        """
        return K.mean(K.square(y_pred - y_true))

    def weighted_mean_squared_error(y_true, y_pred):
        """
        Regularized mean squared error loss. Inspired by the one in mlnet[2].
        Mse_beta is the regularizer.

        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss symbolic value.
        """
        return K.mean(K.square(y_pred - y_true) / (255 - y_true + mse_beta))  # TODO does 255-y_true make sense?

    def sum_squared_errors(y_true, y_pred):
        """
        Sum of squared errors loss.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss symbolic value.
        """
        return K.sum(K.square(y_pred - y_true))

    def kullback_leibler_divergence(y_true, y_pred, eps=K.epsilon()):
        """
        Kullback-Leiber divergence (sec 4.2.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.

        :param y_true: groundtruth.
        :param y_pred: prediction.
        :param eps: regularization epsilon.
        :return: loss value (one symbolic value per batch element).
        """
        P = y_pred
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
        Q = y_true
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        kld = K.sum(Q * K.log(eps + Q/(eps + P)), axis=[1, 2, 3])

        return kld

    def information_gain(y_true, y_pred, y_base, eps=K.epsilon()):
        """
        Information gain (sec 4.1.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.

        :param y_true: groundtruth.
        :param y_pred: prediction.
        :param y_base: baseline.
        :param eps: regularization epsilon.
        :return: loss value (one symbolic value per batch element).
        """
        P = y_pred
        P = P / (K.epsilon() + K.max(P, axis=[1, 2, 3], keepdims=True))
        Q = y_true
        B = y_base

        Qb = K.round(Q)  # discretize at 0.5
        N = K.sum(Qb, axis=[1, 2, 3], keepdims=True)

        ig = K.sum(Qb*(K.log(eps + P) / K.log(2) - K.log(eps + B) / K.log(2)), axis=[1, 2, 3]) / (K.epsilon() + N)

        return ig

    def normalized_scanpath_saliency(y_true, y_pred):
        """
        Normalized Scanpath Saliency (sec 4.1.2 of [1]). Assumes shape (b, 1, h, w) for all tensors.

        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value (one symbolic value per batch element).
        """
        P = y_pred
        P = P / (K.epsilon() + K.max(P, axis=[1, 2, 3], keepdims=True))
        Q = y_true

        Qb = K.round(Q)  # discretize at 0.5
        N = K.sum(Qb, axis=[1, 2, 3], keepdims=True)

        mu_P = K.mean(P, axis=[1, 2, 3], keepdims=True)
        std_P = K.std(P, axis=[1, 2, 3], keepdims=True)
        P_sign = (P - mu_P) / (K.epsilon() + std_P)

        nss = (P_sign * Qb) / (K.epsilon() + N)
        nss = K.sum(nss, axis=[1, 2, 3])

        return -nss  # maximize nss

    def simo_loss(y_true, y_pred):
        """
        Loss defined by simo. Assumes shape (b, 2, h, w) for all tensors.
        y[:, 0, :, :] is saliency, we want KLD for saliency.
        y[:, 1, :, :] is fixation, we want IG for fixation using saliency groundtruth as baseline.

        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value (one symbolic value per batch element).
        """

        y_true_sal = y_true[:, 0:1, :, :]
        y_true_fix = y_true[:, 1:, :, :]

        y_pred_sal = y_pred[:, 0:1, :, :]
        y_pred_fix = y_pred[:, 1:, :, :]

        return kullback_leibler_divergence(y_true_sal, y_pred_sal) - \
               information_gain(y_true_fix, y_pred_fix, y_true_sal)  # maximize information gain over baseline

    if name == 'mse' and mse_beta is not None:
        return weighted_mean_squared_error
    elif name == 'mse' and mse_beta is None:
        return mean_squared_error
    elif name == 'sse':
        return sum_squared_errors
    elif name == 'nss':
        return normalized_scanpath_saliency
    elif name == 'simo':
        return simo_loss
    elif name == 'kld':
        return kullback_leibler_divergence

"""
REFERENCES:
[1] @article{salMetrics_Bylinskii,
  title     = {What do different evaluation metrics tell us about saliency models?},
  author    = {Zoya Bylinskii and Tilke Judd and Aude Oliva and Antonio Torralba and Fr{\'e}do Durand},
  journal   = {arXiv preprint arXiv:1604.03605},
  year      = {2016}
}

[2] https://github.com/marcellacornia/mlnet/blob/master/model.py
"""
