from models import DreyeveNet, saliency_loss, SimpleSaliencyModel
from batch_generators import generate_dreyeve_I_batch, generate_dreyeve_OF_batch, generate_dreyeve_SEG_batch
from batch_generators import generate_dreyeve_batch
from config import batchsize, frames_per_seq, h, w


def fine_tuning():
    model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    model.compile(optimizer='adam', loss=saliency_loss())
    model.summary()

    model.fit_generator(generator=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        samples_per_epoch=batchsize * 32, nb_epoch=4)


def train_image_branch():
    model = SimpleSaliencyModel(input_shape=(3, frames_per_seq, h, w), branch='image')
    model.compile(optimizer='adam', loss=saliency_loss())
    model.summary()

    model.fit_generator(generator=generate_dreyeve_I_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        samples_per_epoch=batchsize * 32, nb_epoch=4)


def train_flow_branch():
    model = SimpleSaliencyModel(input_shape=(3, frames_per_seq, h, w), branch='flow')
    model.compile(optimizer='adam', loss=saliency_loss())
    model.summary()

    model.fit_generator(generator=generate_dreyeve_OF_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        samples_per_epoch=batchsize * 32, nb_epoch=4)

def train_seg_branch():
    model = SimpleSaliencyModel(input_shape=(19, frames_per_seq, h, w), branch='semseg')
    model.compile(optimizer='adam', loss=saliency_loss())
    model.summary()

    model.fit_generator(generator=generate_dreyeve_SEG_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        samples_per_epoch=batchsize * 32, nb_epoch=4)



if __name__ == '__main__':
    train_seg_branch()
