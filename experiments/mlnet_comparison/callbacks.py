import numpy as np
import cv2

import os
from keras.callbacks import Callback, EarlyStopping
from os.path import join

from keras_dl_modules.custom_keras_extensions.callbacks import Checkpointer

from batch_generators import load_batch
from config import experiment_id, batchsize
from config import shape_r, shape_c

from utils import postprocess_predictions

from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import normalize


class PredictionCallback(Callback):
    """
    Callback to perform some debug predictions, on epoch end.
    Loads a batch, predicts it and saves images in `predictions/${experiment_id}`.

    :param experiment_id: the experiment id.
    """

    def __init__(self, experiment_id):

        super(PredictionCallback, self).__init__()

        # create output directories if not existent
        out_dir_path = join('predictions', '{}'.format(experiment_id))
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # set out dir as attribute of PredictionCallback
        self.out_dir_path = out_dir_path

    def on_train_begin(self, logs=None):
            self.on_epoch_end(epoch='begin', logs=logs)

    def on_epoch_end(self, epoch, logs={}):

        # create the folder for predictions
        cur_out_dir = join(self.out_dir_path, 'epoch_{:03d}'.format(epoch) if epoch != 'begin' else 'begin')
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)

        # load and predict
        X, Y = load_batch(batchsize=batchsize, mode='val', gt_type='fix')
        P = self.model.predict(X)

        for b in range(0, batchsize):
            x = X[b].transpose(1, 2, 0)
            x = normalize(x)

            p = postprocess_predictions(P[b, 0], shape_r, shape_c)
            p = np.tile(np.expand_dims(p, axis=2), reps=(1, 1, 3))

            y = postprocess_predictions(Y[b, 0], shape_r, shape_c)
            y = np.tile(np.expand_dims(y, axis=2), reps=(1, 1, 3))
            y = normalize(y)

            # stitch and save
            stitch = stitch_together([x, p, y], layout=(1, 3))
            cv2.imwrite(join(cur_out_dir, '{:02d}.png'.format(b)), stitch)





def get_callbacks():
    """
    Function that returns the list of desired Keras callbacks.
    :return: a list of callbacks.
    """

    return [EarlyStopping(patience=5),
            Checkpointer(join('checkpoints', '{}'.format(experiment_id),
                              'weights.mlnet.{epoch:02d}-{val_loss:.4f}.pkl'),
                         save_best_only=True),
            PredictionCallback(experiment_id)
            ]
