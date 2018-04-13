import argparse
import uuid

from config import batchsize, frames_per_seq, h, w, opt, train_samples_per_epoch, val_samples_per_epoch, nb_epochs
from config import full_frame_loss, crop_loss, w_loss_fine, w_loss_cropped
from batch_generators import generate_dreyeve_I_batch, generate_dreyeve_OF_batch, generate_dreyeve_SEG_batch
from batch_generators import generate_dreyeve_batch
from models import DreyeveNet, SaliencyBranch
from loss_functions import saliency_loss
from callbacks import get_callbacks


def fine_tuning():
    """
    Function to launch training on DreyeveNet. It is called `fine_tuning` since supposes
    the three branches to be pretrained. Should also work from scratch.
    """

    experiment_id = 'DREYEVE_{}'.format(uuid.uuid4())

    model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    model.compile(optimizer=opt,
                  loss={'prediction_fine': saliency_loss(name=full_frame_loss),
                        'prediction_crop': saliency_loss(name=crop_loss)},
                  loss_weights={'prediction_fine': w_loss_fine,
                                'prediction_crop': w_loss_cropped})
    model.summary()

    model.fit_generator(generator=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                               image_size=(h, w), mode='val'),
                        nb_val_samples=val_samples_per_epoch,
                        samples_per_epoch=train_samples_per_epoch,
                        nb_epoch=nb_epochs,
                        callbacks=get_callbacks(experiment_id=experiment_id))


def train_image_branch():
    """
    Function to train a SaliencyBranch model on images.
    """

    experiment_id = 'COLOR_{}'.format(uuid.uuid4())

    model = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='image')
    model.compile(optimizer=opt,
                  loss={'prediction_fine': saliency_loss(name=full_frame_loss),
                        'prediction_crop': saliency_loss(name=crop_loss)},
                  loss_weights={'prediction_fine': w_loss_fine,
                                'prediction_crop': w_loss_cropped})
    model.summary()

    model.fit_generator(generator=generate_dreyeve_I_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                           image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_I_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                                 image_size=(h, w), mode='val'),
                        nb_val_samples=val_samples_per_epoch,
                        samples_per_epoch=train_samples_per_epoch,
                        nb_epoch=nb_epochs,
                        callbacks=get_callbacks(experiment_id=experiment_id))


def train_flow_branch():
    """
    Function to train a SaliencyBranch model on optical flow.
    """
    experiment_id = 'FLOW_{}'.format(uuid.uuid4())

    model = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='flow')
    model.compile(optimizer=opt,
                  loss={'prediction_fine': saliency_loss(name=full_frame_loss),
                        'prediction_crop': saliency_loss(name=crop_loss)},
                  loss_weights={'prediction_fine': w_loss_fine,
                                'prediction_crop': w_loss_cropped})
    model.summary()

    model.fit_generator(generator=generate_dreyeve_OF_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                            image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_OF_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                                  image_size=(h, w), mode='val'),
                        nb_val_samples=val_samples_per_epoch,
                        samples_per_epoch=train_samples_per_epoch,
                        nb_epoch=nb_epochs,
                        callbacks=get_callbacks(experiment_id=experiment_id))


def train_seg_branch():
    """
    Function to train a SaliencyBranch model on semantic segmentation.
    """

    experiment_id = 'SEGM_{}'.format(uuid.uuid4())

    model = SaliencyBranch(input_shape=(19, frames_per_seq, h, w), c3d_pretrained=False, branch='semseg')
    model.compile(optimizer=opt,
                  loss={'prediction_fine': saliency_loss(name=full_frame_loss),
                        'prediction_crop': saliency_loss(name=crop_loss)},
                  loss_weights={'prediction_fine': w_loss_fine,
                                'prediction_crop': w_loss_cropped})
    model.summary()

    model.fit_generator(generator=generate_dreyeve_SEG_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                             image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_SEG_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                                   image_size=(h, w), mode='val'),
                        nb_val_samples=val_samples_per_epoch,
                        samples_per_epoch=train_samples_per_epoch,
                        nb_epoch=nb_epochs,
                        callbacks=get_callbacks(experiment_id=experiment_id))


# training entry point
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--which_branch")

    args = parser.parse_args()
    assert args.which_branch in ['finetuning', 'image', 'flow', 'seg']

    if args.which_branch == 'image':
        train_image_branch()
    elif args.which_branch == 'flow':
        train_flow_branch()
    elif args.which_branch == 'seg':
        train_seg_branch()
    else:
        fine_tuning()
