"""
This script holds the main code used for the training of RMDN.
"""

from batch_generators import generate_RMDN_batch

from models import RMDN

from config import C, batchsize, T, encoding_dim, lr, h, w, hidden_states
from keras.optimizers import RMSprop
from objectives import MDN_neg_log_likelyhood


if __name__ == '__main__':

    model = RMDN(hidden_states=hidden_states, n_mixtures=C, input_shape=(T, encoding_dim), mode='train')
    model.compile(optimizer=RMSprop(lr=lr), loss=MDN_neg_log_likelyhood(image_size=(h, w), B=batchsize, T=T, C=C))

    model.fit_generator(generator=generate_RMDN_batch(batchsize, mode='train'),
                        nb_epoch=1,
                        samples_per_epoch=999*batchsize)
