from dilation import DilationNet, predict
from os.path import join
from glob import glob
import matplotlib.pyplot as plt
import argparse
import os.path
import cv2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence")
    args = parser.parse_args()

    assert args.sequence is not None, 'Please provide a correct sequence number'

    data_dir = 'Z:/DATA/{:02d}/frames/'.format(int(args.sequence))  # local
    # data_dir = '...'  # cineca
    assert os.path.exists(data_dir), 'Assertion error: path {} does not exist'.format(data_dir)

    image_list = glob(join(data_dir, '*.jpg'))

    # get the model
    ds = 'cityscapes'
    model = DilationNet(dataset=ds)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    for img in image_list:
        # read and predict a image
        im = cv2.imread(img)
        y_img = predict(im, model, ds)

        # plot results
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_title('Image')
        a = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(y_img)
        a.set_title('Semantic segmentation')
        plt.show(fig)
