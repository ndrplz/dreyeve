import keras.callbacks
import numpy as np
import cv2
import os

from batch_generators import dreyeve_I_batch, dreyeve_OF_batch, dreyeve_SEG_batch, dreyeve_batch
from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import write_image, normalize
from config import batchsize, frames_per_seq, h, w
from keras.callbacks import ReduceLROnPlateau
from utils import seg_to_colormap
from os.path import join


class Checkpointer(keras.callbacks.Callback):
    def __init__(self, branch):
        # create output directories if not existent
        out_dir_path = join('checkpoints', branch)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # set out dir as attribute of PredictionCallback
        self.out_dir_path = out_dir_path

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(join(self.out_dir_path, 'w.h5'))


# TODO make this work for finetuning
class PredictionCallback(keras.callbacks.Callback):

    def __init__(self, branch):

        self.branch = branch

        # create output directories if not existent
        out_dir_path = join('predictions', branch)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # set out dir as attribute of PredictionCallback
        self.out_dir_path = out_dir_path

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 0:
            self.on_epoch_end(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):

        if self.branch == 'image':
            X, Y = dreyeve_I_batch(batchsize=2 * batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                   mode='val', gt_type='fix')
        elif self.branch == 'optical_flow':
            X, Y = dreyeve_OF_batch(batchsize=2 * batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                    mode='val', gt_type='fix')
        elif self.branch == 'semseg':
            X, Y = dreyeve_SEG_batch(batchsize=2 * batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                     mode='val', gt_type='fix')
        elif self.branch == 'all':
            X, Y = dreyeve_batch(batchsize=2 * batchsize, nb_frames=frames_per_seq, image_size=(h, w),
                                 mode='val', gt_type='fix')

        # predict batch
        Z = self.model.predict(X)

        for b in range(0, 2 * batchsize):
            # image
            if self.branch == 'image':
                x_img = X[0][b]  # fullframe, b-th image
                x_img = np.squeeze(x_img, axis=1).transpose(1, 2, 0)
            elif self.branch == 'optical_flow':
                x_img = X[0][b]  # fullframe, b-th image
                x_img = np.squeeze(x_img, axis=1).transpose(1, 2, 0)
            elif self.branch == 'semseg':
                x_img = X[0][b]  # fullframe, b-th image
                x_img = seg_to_colormap(np.argmax(np.squeeze(x_img, axis=1), axis=0))

            # prediction
            z_img = np.tile(np.expand_dims(Z[0][b, 0] * 255, axis=2), reps=(1, 1, 3)).astype(np.uint8)

            # groundtruth
            y_img = np.tile(np.expand_dims(Y[0][b, 0] * 255, axis=2), reps=(1, 1, 3)).astype(np.uint8)

            # stitch and write
            stitch = stitch_together([normalize(x_img), z_img, y_img], layout=(1, 3))
            write_image(join(self.out_dir_path, 'e{:02d}_{:02d}.png'.format(epoch+1, b+1)), stitch,
                        normalize=True, channels_first=False)


def get_callbacks(branch):
    assert branch in ['image', 'optical_flow', 'semseg', 'all'], 'Unknown model {} for get callbacks'.format(branch)

    return [PredictionCallback(branch=branch),
            Checkpointer(branch=branch),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)]
