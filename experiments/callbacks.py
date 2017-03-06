import keras.callbacks
import numpy as np
import os

from batch_generators import dreyeve_I_batch, dreyeve_OF_batch, dreyeve_SEG_batch, dreyeve_batch
from computer_vision_utils.stitching import stitch_together

from keras.callbacks import ReduceLROnPlateau, CSVLogger
from computer_vision_utils.io_helper import write_image, normalize
from config import frames_per_seq, h, w, log_dir, callback_batchsize
from utils import seg_to_colormap
from os.path import join, exists


def get_branch_from_experiment_id(experiment_id):

    assert isinstance(experiment_id, basestring), "Experiment ID must be a string."

    branch = None
    if experiment_id.lower().startswith('dreyeve'):
        branch = "all"
    elif experiment_id.lower().startswith('color'):
        branch = "image"
    elif experiment_id.lower().startswith('flow'):
        branch = "optical_flow"
    elif experiment_id.lower().startswith('segm'):
        branch = "semseg"

    return branch


class Checkpointer(keras.callbacks.Callback):
    def __init__(self, experiment_id):
        # create output directories if not existent
        out_dir_path = join('checkpoints', experiment_id)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        self.out_dir_path = out_dir_path

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(join(self.out_dir_path, 'w_epoch_{:06d}.h5'.format(epoch)))


# TODO make this work for finetuning
class PredictionCallback(keras.callbacks.Callback):

    def __init__(self, experiment_id):

        self.branch = get_branch_from_experiment_id(experiment_id)

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

        if self.branch == 'image':
            # X is [B_ff, B_s, B_c]
            X, Y = dreyeve_I_batch(batchsize=callback_batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                   mode='val', gt_type='fix')
        elif self.branch == 'optical_flow':
            X, Y = dreyeve_OF_batch(batchsize=callback_batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                    mode='val', gt_type='fix')
        elif self.branch == 'semseg':
            X, Y = dreyeve_SEG_batch(batchsize=callback_batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                     mode='val', gt_type='fix')
        elif self.branch == 'all':
            X, Y = dreyeve_batch(batchsize=callback_batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                 mode='val', gt_type='fix')

        # predict batch
        Z = self.model.predict(X)

        for b in range(0, callback_batchsize):
            # image
            if self.branch == 'image':

                x_ff_img = X[0][b]  # fullframe, b-th image
                x_ff_img = np.squeeze(x_ff_img, axis=1).transpose(1, 2, 0)

            elif self.branch == 'optical_flow':
                x_ff_img = X[0][b]  # fullframe, b-th image
                x_ff_img = np.squeeze(x_ff_img, axis=1).transpose(1, 2, 0)
            elif self.branch == 'semseg':
                x_ff_img = X[0][b]  # fullframe, b-th image
                x_ff_img = seg_to_colormap(np.argmax(np.squeeze(x_ff_img, axis=1), axis=0), channels_first=True)

            x_sm_img = X[1][b][:, -1, :, :]  # resized frame (last one), b-th image
            x_sm_img = x_sm_img.transpose(1, 2, 0)

            x_cr_img = X[2][b][:, -1, :, :]  # cropped frame (last one), b-th image
            x_cr_img = x_cr_img.transpose(1, 2, 0)

            # prediction
            z_ff_img = np.tile(np.expand_dims(normalize(Z[0][b, 0]), axis=2), reps=(1, 1, 3)).astype(np.uint8)
            z_cr_img = np.tile(np.expand_dims(normalize(Z[1][b, 0]), axis=2), reps=(1, 1, 3)).astype(np.uint8)

            # groundtruth
            y_ff_img = np.tile(np.expand_dims(normalize(Y[0][b, 0]), axis=2), reps=(1, 1, 3))
            y_cr_img = np.tile(np.expand_dims(normalize(Y[1][b, 0]), axis=2), reps=(1, 1, 3))

            # stitch and write
            stitch_ff = stitch_together([normalize(x_ff_img), z_ff_img, y_ff_img], layout=(1, 3), resize_dim=(500, 1500))
            stitch_cr = stitch_together([normalize(x_cr_img), z_cr_img, y_cr_img], layout=(1, 3), resize_dim=(500, 1500))
            write_image(join(self.out_dir_path, 'ff_e{:02d}_{:02d}.png'.format(epoch + 1, b + 1)), stitch_ff, channels_first=False)
            write_image(join(self.out_dir_path, 'cr_e{:02d}_{:02d}.png'.format(epoch + 1, b + 1)), stitch_cr, channels_first=False)


def get_callbacks(experiment_id):

    if not exists(log_dir):
        os.makedirs(log_dir)

    return [PredictionCallback(experiment_id=experiment_id),
            Checkpointer(experiment_id=experiment_id),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),
            CSVLogger(join(log_dir, '{}.txt'.format(experiment_id)))]

