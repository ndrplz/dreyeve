import numpy as np
import os

from keras.callbacks import ReduceLROnPlateau, Callback
from os.path import join, exists

from config import DREYEVE_ROOT
from config import frames_per_seq, h, w, batchsize, T

from batch_generators import RMDN_batch

from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import read_image, write_image, normalize

from keras_dl_modules.custom_keras_extensions.callbacks import Checkpointer

from utils import gmm_to_probability_map


class ModelLoader(Callback):
    """
    Callback to load weights into the network at the beginning of training.

    :param h5_file: the weight file to load.
    """
    def __init__(self, h5_file=None):

        super(ModelLoader, self).__init__()

        self.h5_file = h5_file

    def on_train_begin(self, logs={}):
        if self.h5_file is not None:
            self.model.load_weights(self.h5_file)


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

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 0:
            self.on_epoch_end(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):

        # create epoch folder
        epoch_out_dir = join(self.out_dir_path, '{:02d}'.format(epoch))
        if not exists(epoch_out_dir):
            os.makedirs(epoch_out_dir)

        # load batch
        seqs, frs, X, Y = RMDN_batch(batchsize=batchsize, mode='val')

        # predict batch
        Z = self.model.predict(X)

        for b in range(0, batchsize):
            for t in range(0, T):
                seq = seqs[b, t]
                id = frs[b, t]

                # load this frame image and gt
                rgb_frame = read_image(join(DREYEVE_ROOT, '{:02d}'.format(seq), 'frames', '{:06d}.jpg'.format(id)),
                                       channels_first=False, resize_dim=(h, w), dtype=np.uint8)
                fixation_map = read_image(join(DREYEVE_ROOT, '{:02d}'.format(seq), 'saliency_fix',
                                               '{:06d}.png'.format(id+1)),
                                       channels_first=False, resize_dim=(h, w), dtype=np.uint8)

                # extract this frame mixture
                gmm = Z[b, t]

                pred_map = gmm_to_probability_map(gmm=gmm, image_size=(h, w))
                pred_map = normalize(pred_map)
                pred_map = np.tile(np.expand_dims(pred_map, axis=-1), reps=(1, 1, 3))

                # stitch
                stitch = stitch_together([rgb_frame, pred_map, fixation_map], layout=(1, 3))
                write_image(join(epoch_out_dir, '{:02d}_{:02d}.jpg'.format(b, t)), stitch)



def get_callbacks(experiment_id):
    """
    Helper function to build the list of desired keras callbacks.

    :param experiment_id: experiment id.
    :return: a list of Keras Callbacks.
    """

    return [
            # ModelLoader(h5_file=None),
            PredictionCallback(experiment_id=experiment_id),
            Checkpointer(join('rmdn_checkpoints', experiment_id, 'weights.{epoch:02d}')),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
