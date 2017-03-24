from __future__ import division
from keras.optimizers import SGD

from callbacks import get_callbacks

from model import ml_net_model, loss
from batch_generators import generate_batch

from config import shape_c, shape_r, batchsize
from config import nb_samples_per_epoch, nb_epoch, nb_imgs_val


if __name__ == '__main__':

    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(sgd, loss)
    model.summary()

    print("Training ML-Net")
    model.fit_generator(generator=generate_batch(batchsize=batchsize,
                                                 mode='train',
                                                 gt_type='fix'),
                        validation_data=generate_batch(batchsize=batchsize,
                                                       mode='val',
                                                       gt_type='fix'),
                        nb_val_samples=nb_imgs_val,
                        nb_epoch=nb_epoch,
                        samples_per_epoch=nb_samples_per_epoch,
                        callbacks=get_callbacks())
