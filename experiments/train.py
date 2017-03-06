from models import DreyeveNet, saliency_loss, SaliencyBranch, FlavorBranch
from batch_generators import generate_dreyeve_I_batch, generate_dreyeve_OF_batch, generate_dreyeve_SEG_batch
from batch_generators import generate_dreyeve_batch
from config import batchsize, frames_per_seq, h, w, opt, full_frame_loss, crop_loss, w_loss_fine, w_loss_cropped
import uuid
from callbacks import get_callbacks


def fine_tuning():

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
                        nb_val_samples=batchsize * 5,
                        samples_per_epoch=batchsize * 256,
                        nb_epoch=999,
                        callbacks=get_callbacks(experiment_id=experiment_id))


def train_image_branch():

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
                        nb_val_samples=batchsize * 5,
                        samples_per_epoch=batchsize * 64,
                        nb_epoch=999,
                        callbacks=get_callbacks(experiment_id=experiment_id))


def train_flow_branch():

    experiment_id = 'FLOW_{}'.format(uuid.uuid4())

    model = FlavorBranch(input_shape=(3, frames_per_seq, h, w), branch='flow')
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
                        nb_val_samples=batchsize * 5,
                        samples_per_epoch=batchsize * 64,
                        nb_epoch=999,
                        callbacks=get_callbacks(experiment_id=experiment_id))


def train_seg_branch():

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
                        nb_val_samples=batchsize*5,
                        samples_per_epoch=batchsize * 256,
                        nb_epoch=999,
                        callbacks=get_callbacks(experiment_id=experiment_id))


if __name__ == '__main__':
    train_flow_branch()
