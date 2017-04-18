import keras.backend as K


def MDN_neg_log_likelyhood(image_size, B, T, C):
    """
    This function returns the negative log likelyhood loss from the original paper.

    :param image_size: target fixation map size (h, w).
    :param B: batchsize.
    :param T: number of temporal samples per batch
    :param C: number of mixtures in the output GMM.
    :return: loss symbolic value (one per batch element).
    """
    h, w = image_size

    def loss(y_true, gmm_pred):
        """
        :param y_true: has shape (batchsize, timesteps, h, w).
        :param gmm_pred: has shape (batchsize, timesteps, n_mixtures, 6).
        """
        # from config import h, w, C, T
        # from config import batchsize as B

        pi = 3.14159265359

        weight = gmm_pred[:, :, :, :1]  # (B, T, C, 1)
        mu = gmm_pred[:, :, :, 1:3]  # (B, T, C, 2)
        sigma = gmm_pred[:, :, :, 3:5]  # (B, T, C, 2)
        sigma = K.expand_dims(sigma)  # (B, T, C, 2, 1)
        ro = gmm_pred[:, :, :, 5:]  # (B, T, C, 1)
        ro = K.expand_dims(ro)  # (B, T, C, 1, 1)

        row_idx = K.repeat_elements(K.expand_dims(K.arange(start=0, stop=h, step=1, dtype='float32'), dim=1),
                                    rep=w, axis=1)
        col_idx = K.repeat_elements(K.expand_dims(K.arange(start=0, stop=w, step=1, dtype='float32'), dim=0),
                                    rep=h, axis=0)

        # concatenate to get features
        row_idx = K.expand_dims(row_idx, dim=0)
        col_idx = K.expand_dims(col_idx, dim=0)
        x = K.concatenate([row_idx, col_idx], axis=0)  # (2, h, w)
        x = K.reshape(x, shape=(2, h*w))

        # print 'X: {}'.format(K.eval(x))

        # expand to B, T, C
        x = K.reshape(x, shape=(1, 1, 1, 2, h*w))

        # generate determinant tensor
        det = K.prod(sigma, axis=3, keepdims=True) - K.square(ro)  # (B, T, C, 1, 1)
        # print 'det: {}'.format(K.eval(det))

        # evaluating x-mu
        x_minus_mu = x - K.expand_dims(mu)  # (B, T, C, 2, h*w)

        x_minus_mu_dp = K.prod(x_minus_mu, axis=3, keepdims=True)
        x_minus_mu_sq = K.square(x_minus_mu)

        # mega formula of death
        esp = -(x_minus_mu_sq[:, :, :, :1, :] * sigma[:, :, :, 1:, :]
                - 2 * x_minus_mu_dp * ro
                + x_minus_mu_sq[:, :, :, 1:, :] * sigma[:, :, :, :1, :]) / (K.epsilon() + 2 * det)
        # print 'esp: {}'.format(K.eval(esp))

        gauss = K.exp(esp) / (K.epsilon() + 2 * pi * K.sqrt(det))
        # print 'gauss: {}'.format(K.eval(gauss))

        # weight with mixture component
        mixture = K.expand_dims(weight) * gauss

        # apply log
        neg_lkl = - K.log(K.epsilon() + K.sum(mixture, axis=2, keepdims=True))
        # print 'neg_lkl: {}'.format(K.eval(neg_lkl))

        # multiply with fixation probability
        y_true /= (K.epsilon() + K.sum(y_true, axis=[2, 3], keepdims=True))
        y_true = K.reshape(y_true, shape=(B, T, 1, 1, h*w))

        return K.sum((y_true*neg_lkl), axis=[1, 2, 3, 4])

    return loss


# if __name__ == '__main__':
#     import numpy as np
#     y_true_num = np.load('Y.npy')
#     gmm_pred_num = np.load('P.npy')
#
#     loss(y_true=K.variable(y_true_num), gmm_pred=K.variable(gmm_pred_num))
