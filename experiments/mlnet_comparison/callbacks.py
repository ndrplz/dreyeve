import os
from keras.callbacks import Callback, EarlyStopping
from os.path import join

from keras_dl_modules.custom_keras_extensions.callbacks import Checkpointer

from batch_generators import load_batch
from config import experiment_id, batchsize

from utils import postprocess_predictions


class PredictionCallback(Callback):
    """
    Callback to perform some debug predictions, on epoch end.
    Loads a batch, predicts it and saves images in `predictions/${experiment_id}`.

    :param experiment_id: the experiment id.
    """

    def __init__(self, experiment_id):

        super(PredictionCallback, self).__init__()

        # create output directories if not existent
        out_dir_path = join('predictions', experiment_id)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # set out dir as attribute of PredictionCallback
        self.out_dir_path = out_dir_path

    def on_train_begin(self, logs=None):
            self.on_epoch_end(epoch='begin', logs=logs)

    def on_epoch_end(self, epoch, logs={}):

        # load and predict
        X, Y = load_batch(batchsize=batchsize, mode='val', gt_type='fix')
        P = self.model.predict(X)

        P = postprocess_predictions(P)
        for b in range(0, batchsize):

            x = X[0].transpose(1, 2, 0)
            p = P[0]
            y = Y[0]

        # TODO complete this callback


def get_callbacks():
    """
    Function that returns the list of desired Keras callbacks.
    :return: a list of callbacks.
    """

    return [EarlyStopping(patience=5),
            Checkpointer(join('checkpoints','{}'.format(experiment_id), 'weights.mlnet.{epoch:02d}-{val_loss:.4f}.pkl'),
                         save_best_only=True)
            ]
