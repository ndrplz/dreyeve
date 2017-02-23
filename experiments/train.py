from models import DreyeveNet, saliency_loss
from batch_generators import generate_dreyeve_batch

if __name__ == '__main__':
    batchsize = 1
    frames_per_seq, h, w = 16, 256, 512

    model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    model.compile(optimizer='adam', loss=saliency_loss())
    model.summary()

    model.fit_generator(generator=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
                                                         image_size=(h, w), mode='train'),
                        samples_per_epoch=batchsize*32, nb_epoch=4)
