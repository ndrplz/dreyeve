from dilation import DilationNet, predict_no_tiles
from os.path import join
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import os.path
import cv2

# cityscapes dataset palette
palette = np.array([[128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [70, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32]], dtype='uint8')


def seg_to_colormap(seg, channels_first):
    """
    Function to turn segmentation PREDICTION (not probabilities) to colormap.

    :param seg: the prediction image, having shape (h,w)
    :param channels_first: if true, returns (c,h,w) rather than (h,w,c)
    :return: the colormap image, having shape (h,w,3)
    """
    h, w = seg.shape
    color_image = palette[seg.ravel()].reshape(h, w, 3)

    if channels_first:
        color_image = color_image.transpose(2, 0, 1)

    return color_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence")
    args = parser.parse_args()

    assert args.sequence is not None, 'Please provide a correct sequence number'

    dreyeve_dir = '/WindowsShares/F/DREYEVE/DATA/'  # local
    #dreyeve_dir = '/gpfs/work/IscrC_DeepVD/dabati/DREYEVE/data/'  # cineca

    data_dir = dreyeve_dir + '{:02d}/frames'.format(int(args.sequence))  # local
    out_dir = 'maserati_semseg/{:02d}/'.format(int(args.sequence))  # local
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

        y_img = seg_to_colormap(np.argmax(y[0], axis=0), channels_first=False)
        y_img = cv2.cvtColor(y_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(join(out_dir, os.path.basename(img)), y_img)


