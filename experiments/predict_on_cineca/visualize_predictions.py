import numpy as np
import cv2

from random import choice

from os.path import join

from train.config import frames_per_seq, h, w, total_frames_each_run
from train.config import dreyeve_dir
from train.config import dreyeve_test_seq, test_frame_range
from train.utils import seg_to_colormap

from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.io_helper import normalize, read_image

pred_dir = 'Z:/PREDICTIONS_2017'


def read_lines_from_file(filename):
    """
    Function to read lines from file

    :param filename: The text file to be read.
    :return: content: A list of strings
    """
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content


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


def sample_signature(sequences, allowed_frames, force_sample_steering):
    """
    Function to create a unique batch signature for the Dreyeve dataset, for visualization.

    :param sequences: sequences to sample from.
    :param allowed_frames: range of allowed frames to sample the sequence start from.
    :param force_sample_steering: whether or not to sample sequences while the car is steering.
    :return: a tuple like (num_run, start).
    """
    # get random sequence
    num_run = choice(sequences)

    # get random start of sequence
    p = np.ones(total_frames_each_run)
    mask = np.zeros(total_frames_each_run)
    mask[np.array(allowed_frames)] = 1
    if force_sample_steering:
        steering_dir_file = join(dreyeve_dir, '{:02d}'.format(num_run), 'steering_directions.txt')
        steering_dirs = read_lines_from_file(steering_dir_file)
        prob_straight = 1 - float(len([s for s in steering_dirs if s == 'STRAIGHT'])) / len(steering_dirs)
        prob_left = 1 - float(len([s for s in steering_dirs if s == 'LEFT'])) / len(steering_dirs)
        prob_right = 1 - float(len([s for s in steering_dirs if s == 'RIGHT'])) / len(steering_dirs)
        p[[i for i in xrange(0, len(steering_dirs)) if steering_dirs[i] == 'STRAIGHT']] = prob_straight
        p[[i for i in xrange(0, len(steering_dirs)) if steering_dirs[i] == 'LEFT']] = prob_left
        p[[i for i in xrange(0, len(steering_dirs)) if steering_dirs[i] == 'RIGHT']] = prob_right
    p += 1e-10
    p *= mask
    p /= np.sum(p)

    start = np.random.choice(range(0, total_frames_each_run), p=p)

    return num_run, start


def load(seq, frame):
    """
    Function to load a 16-frames sequence to plot

    :param seq: the sequence number
    :param frame: the frame inside the sequence
    :return: a stitched image to show
    """

    small_size = (270, 480)

    sequence_x_dir = join(dreyeve_dir, '{:02d}'.format(seq))
    sequence_y_dir = join(pred_dir, '{:02d}'.format(seq))

    # x
    x_img = read_image(join(sequence_x_dir, 'frames', '{:06d}.jpg'.format(frame)), channels_first=False,
                       color_mode='BGR', dtype=np.uint8)
    x_img_small = cv2.resize(x_img, small_size[::-1])
    of_img = read_image(join(sequence_x_dir, 'optical_flow', '{:06d}.png'.format(frame+1)), channels_first=False,
                       color_mode='BGR', resize_dim=small_size)
    seg_img = seg_to_colormap(np.argmax(np.squeeze(
                        np.load(join(sequence_x_dir, 'semseg', '{:06d}.npz'.format(frame)))['arr_0']), axis=0),
                            channels_first=False)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)

    # pred
    image_p = normalize(np.squeeze(np.load(join(sequence_y_dir, 'image_branch', '{:06d}.npz'.format(frame)))['arr_0']))
    flow_p = normalize(np.squeeze(np.load(join(sequence_y_dir, 'flow_branch', '{:06d}.npz'.format(frame)))['arr_0']))
    semseg_p = normalize(np.squeeze(np.load(join(sequence_y_dir, 'semseg_branch', '{:06d}.npz'.format(frame)))['arr_0']))
    dreyevenet_p = normalize(np.squeeze(np.load(join(sequence_y_dir, 'dreyeveNet', '{:06d}.npz'.format(frame)))['arr_0']))

    image_p = cv2.resize(cv2.cvtColor(image_p, cv2.COLOR_GRAY2BGR), small_size[::-1])
    flow_p = cv2.resize(cv2.cvtColor(flow_p, cv2.COLOR_GRAY2BGR), small_size[::-1])
    semseg_p = cv2.resize(cv2.cvtColor(semseg_p, cv2.COLOR_GRAY2BGR), small_size[::-1])
    dreyevenet_p = cv2.resize(cv2.cvtColor(dreyevenet_p, cv2.COLOR_GRAY2BGR), small_size[::-1])

    s1 = stitch_together([x_img_small, of_img, seg_img, image_p, flow_p, semseg_p], layout=(2, 3))
    x_img = cv2.resize(x_img, dsize=(s1.shape[1], s1.shape[0]))
    dreyevenet_p = cv2.resize(dreyevenet_p, dsize=(s1.shape[1], s1.shape[0]))
    dreyevenet_p = cv2.applyColorMap(dreyevenet_p, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(x_img, 0.5, dreyevenet_p, 0.5, gamma=0)

    stitch = stitch_together([s1, blend], layout=(2, 1), resize_dim=(720, 970))

    return stitch


if __name__ == '__main__':

    sequence_length = 16

    while True:

        # generate a signature
        seq, start_frame = sample_signature(sequences=dreyeve_test_seq, allowed_frames=range(15, 7500-1),
                                            force_sample_steering=True)

        for offset in xrange(0, sequence_length):
            to_show = load(seq=seq, frame=start_frame+offset)

            # visualize
            cv2.imshow('prediction', to_show)
            cv2.waitKey()
