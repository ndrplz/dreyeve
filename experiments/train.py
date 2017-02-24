from models import DreyeveNet, saliency_loss, SimpleSaliencyModel
from batch_generators import generate_dreyeve_I_batch, generate_dreyeve_OF_batch, generate_dreyeve_SEG_batch
from batch_generators import generate_dreyeve_batch
from config import batchsize, frames_per_seq, h, w, opt

from callbacks import get_callbacks


def fine_tuning():
    model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    model.compile(optimizer=opt, loss=saliency_loss(), loss_weights=[1, 100])
    model.summary()

    model.fit_generator(generator=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                               image_size=(h, w), mode='val'),
                        nb_val_samples=batchsize * 5,
                        samples_per_epoch=batchsize * 256,
                        nb_epoch=999,
                        callbacks=get_callbacks(branch='all'))


def train_image_branch():
    model = SimpleSaliencyModel(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='image')
    model.compile(optimizer=opt, loss=saliency_loss(), loss_weights=[1, 100])
    model.summary()

    model.fit_generator(generator=generate_dreyeve_I_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                           image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_I_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                                 image_size=(h, w), mode='val'),
                        nb_val_samples=batchsize * 5,
                        samples_per_epoch=batchsize * 256,
                        nb_epoch=999,
                        callbacks=get_callbacks(branch='image'))


def train_flow_branch():
    model = SimpleSaliencyModel(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='flow')
    model.compile(optimizer=opt, loss=saliency_loss(), loss_weights=[1, 100])
    model.summary()

    model.fit_generator(generator=generate_dreyeve_OF_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                            image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_OF_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                                  image_size=(h, w), mode='val'),
                        nb_val_samples=batchsize * 5,
                        samples_per_epoch=batchsize * 256,
                        nb_epoch=999,
                        callbacks=get_callbacks(branch='optical_flow'))


def train_seg_branch():
    model = SimpleSaliencyModel(input_shape=(19, frames_per_seq, h, w), c3d_pretrained=False, branch='semseg')
    model.compile(optimizer=opt, loss=saliency_loss(), loss_weights=[1, 100])
    model.summary()

    model.fit_generator(generator=generate_dreyeve_SEG_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        validation_data=generate_dreyeve_SEG_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                   image_size=(h, w), mode='val'),
                        nb_val_samples=batchsize*5,
                        samples_per_epoch=batchsize * 256,
                        nb_epoch=999,
                        callbacks=get_callbacks(branch='semseg'))


if __name__ == '__main__':
    train_image_branch()
