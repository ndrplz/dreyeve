"""
This script holds the main code used for the training of RMDN.
"""
import uuid

from batch_generators import generate_RMDN_batch

from models import RMDN_train

from config import C, batchsize, T, encoding_dim, lr, h, w, hidden_states, nb_epoch, samples_per_epoch
from keras.optimizers import RMSprop
from objectives import MDN_neg_log_likelyhood

from callbacks import get_callbacks


if __name__ == '__main__':

    # get a new experiment id
    experiment_id = str(uuid.uuid4())

    model = RMDN_train(hidden_states=hidden_states, n_mixtures=C, input_shape=(T, encoding_dim))
    model.compile(optimizer=RMSprop(lr=lr), loss=MDN_neg_log_likelyhood(image_size=(h, w), B=batchsize, T=T, C=C))

    model.fit_generator(generator=generate_RMDN_batch(batchsize, mode='train'),
                        nb_epoch=nb_epoch,
                        samples_per_epoch=samples_per_epoch,
                        callbacks=get_callbacks(experiment_id))
