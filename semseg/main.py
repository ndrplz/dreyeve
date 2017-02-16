from dilation import DilationNet, predict_no_tiles
from os.path import join
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import os.path
import cv2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence")
    args = parser.parse_args()

    assert args.sequence is not None, 'Please provide a correct sequence number'

    # dreyeve_dir = '/home/aba/dreyeve/data/'  # local
    dreyeve_dir = '/gpfs/work/IscrC_DeepVD/dabati/dreyeve_semantic_seg/data/'  # cineca

    data_dir = dreyeve_dir + '{:02d}/frames'.format(int(args.sequence))  # local
    out_dir = dreyeve_dir + '{:02d}/semseg'.format(int(args.sequence))  # local
    assert os.path.exists(data_dir), 'Assertion error: path {} does not exist'.format(data_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    image_list = glob(join(data_dir, '*.jpg'))

    # get the model
    ds = 'cityscapes'
    model = DilationNet(dataset=ds, input_shape=(3, 1452, 2292))
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()

    for img in tqdm(image_list):
        # read and predict a image
        im = cv2.imread(img)
        y = predict_no_tiles(im, model, ds)
        y_resized = np.zeros(shape=(1, 19, 270, 480))

        for c in range(0, y.shape[1]):
            y_resized[0, c] = cv2.resize(y[0, c], dsize=(480, 270))

        np.savez_compressed(join(out_dir, os.path.basename(img)[0:-4]), y_resized)


