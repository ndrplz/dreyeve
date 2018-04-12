import numpy as np
import os

from keras.callbacks import ReduceLROnPlateau, CSVLogger, Callback
from os.path import join, exists

from config import frames_per_seq, h, w, log_dir, callback_batchsize
from config import ckp_dir, prd_dir
from batch_generators import dreyeve_I_batch, dreyeve_OF_batch, dreyeve_SEG_batch, dreyeve_batch
from utils import seg_to_colormap, get_branch_from_experiment_id

from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import write_image, normalize


class ModelLoader(Callback):
    """
    Callback to load weights into the network at the beginning of training.

    :param experiment_id: the experiment id.
    :param image_h5: path of the .h5 file to load for the image branch.
    :param flow_h5: path of the .h5 file to load for the optical flow branch.
    :param seg_h5: path of the .h5 file to load for the segmentation branch.
    :param all_h5: path of the .h5 file to load for the whole DreyeveNet.
    """

    def __init__(self, experiment_id, image_h5=None, flow_h5=None, seg_h5=None, all_h5=None):

        super(ModelLoader, self).__init__()

        self.branch = get_branch_from_experiment_id(experiment_id)

        self.image_h5 = image_h5
        self.flow_h5 = flow_h5
        self.seg_h5 = seg_h5
        self.all_h5 = all_h5

    def on_train_begin(self, logs={}):
        if self.branch == 'image' and self.image_h5 is not None:
            self.model.load_weights(self.image_h5)
        elif self.branch == 'optical_flow' and self.flow_h5 is not None:
            self.model.load_weights(self.flow_h5)
        elif self.branch == 'semseg' and self.seg_h5 is not None:
            self.model.load_weights(self.seg_h5)
        elif self.branch == 'all':
            if self.all_h5 is not None:
                self.model.load(self.all_h5)
            else:
                if self.image_h5 is not None:
                    m = [l for l in self.model.layers if l.name == 'image_saliency_branch'][0]
                    m.load_weights(self.image_h5)
                if self.flow_h5 is not None:
                    m = [l for l in self.model.layers if l.name == 'optical_flow_saliency_branch'][0]
                    m.load_weights(self.flow_h5)
                if self.seg_h5 is not None:
                    m = [l for l in self.model.layers if l.name == 'segmentation_saliency_branch'][0]
                    m.load_weights(self.seg_h5)


class Checkpointer(Callback):
    """
    Callback to save weights of a model, on epoch end.

    :param experiment_id: the experiment id.
    """
    def __init__(self, experiment_id):

        super(Checkpointer, self).__init__()

        # create output directories if not existent
        out_dir_path = join(ckp_dir, experiment_id)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        self.out_dir_path = out_dir_path

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(join(self.out_dir_path, 'w_epoch_{:06d}.h5'.format(epoch)))


class PredictionCallback(Callback):
    """
    Callback to perform some debug predictions, on epoch end.
    Loads a batch, predicts it and saves images in `predictions/${experiment_id}`.

    :param experiment_id: the experiment id.
    """

    def __init__(self, experiment_id):

        super(PredictionCallback, self).__init__()

        self.branch = get_branch_from_experiment_id(experiment_id)

        # create output directories if not existent
        out_dir_path = join(prd_dir, experiment_id)
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

            if self.branch == 'image':
                x_ff_img = X[0][b]  # fullframe, b-th image
                x_ff_img = np.squeeze(x_ff_img, axis=1).transpose(1, 2, 0)

                x_cr_img = X[2][b][:, -1, :, :]  # cropped frame (last one), b-th image
                x_cr_img = x_cr_img.transpose(1, 2, 0)
            elif self.branch == 'optical_flow':
                x_ff_img = X[0][b]  # fullframe, b-th image
                x_ff_img = np.squeeze(x_ff_img, axis=1).transpose(1, 2, 0)

                x_cr_img = X[2][b][:, -1, :, :]  # cropped frame (last one), b-th image
                x_cr_img = x_cr_img.transpose(1, 2, 0)
            elif self.branch == 'semseg':
                x_ff_img = X[0][b]  # fullframe, b-th image
                x_ff_img = seg_to_colormap(np.argmax(np.squeeze(x_ff_img, axis=1), axis=0), channels_first=False)

                x_cr_img = X[2][b][:, -1, :, :]  # cropped frame (last one), b-th image
                x_cr_img = seg_to_colormap(np.argmax(x_cr_img, axis=0), channels_first=False)
            elif self.branch == 'all':
                # fullframe
                i_ff_img = X[0][b]
                i_ff_img = np.squeeze(i_ff_img, axis=1).transpose(1, 2, 0)
                of_ff_img = X[3][b]
                of_ff_img = np.squeeze(of_ff_img, axis=1).transpose(1, 2, 0)
                seg_ff_img = seg_to_colormap(np.argmax(np.squeeze(X[6][b], axis=1), axis=0), channels_first=False)

                x_ff_img = stitch_together([normalize(i_ff_img), normalize(of_ff_img), normalize(seg_ff_img)],
                                           layout=(3, 1), resize_dim=i_ff_img.shape[:2])  # resize like they're one

                # crop
                i_cr_img = X[2][b][:, -1, :, :].transpose(1, 2, 0)
                of_cr_img = X[5][b][:, -1, :, :].transpose(1, 2, 0)
                seg_cr_img = seg_to_colormap(np.argmax(X[8][b][:, -1, :, :], axis=0), channels_first=False)

                x_cr_img = stitch_together([normalize(i_cr_img), normalize(of_cr_img), normalize(seg_cr_img)],
                                           layout=(3, 1), resize_dim=i_cr_img.shape[:2])  # resize like they're one

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
    """
    Helper function to build the list of desired keras callbacks.

    :param experiment_id: experiment id.
    :return: a list of Keras Callbacks.
    """
    if not exists(log_dir):
        os.makedirs(log_dir)

    return [
            ModelLoader(experiment_id=experiment_id),
            PredictionCallback(experiment_id=experiment_id),
            Checkpointer(experiment_id=experiment_id),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),
            CSVLogger(join(log_dir, '{}.txt'.format(experiment_id)))
    ]

