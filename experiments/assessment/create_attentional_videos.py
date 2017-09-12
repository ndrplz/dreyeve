"""
This script is used to construct videos mimicking attentional behavior of
humans, deep model, central baseline. Such videos will be then used for
the quality assessment (Sec 5.4 of the paper).
"""

import numpy as np
from os.path import join, exists
import os

import skimage.io as io
from skimage.filters import gaussian
from skimage.transform import resize

import uuid
from tqdm import tqdm
import skvideo.io


from scipy.ndimage.morphology import distance_transform_edt

import matplotlib.pyplot as plt
plt.ion()

# parameters
dreyeve_root = '/majinbu/public/DREYEVE'
output_root = '/majinbu/public/DREYEVE/QUALITY_ASSESSMENT_VIDEOS'
output_txt = join(output_root, 'videos.txt')
n_frames = 1000
shape = (1080 // 2, 1920 // 2)
sigma_blur = 50
rbf_sigma = 5


def read_frame(seq, idx):
    """
    Reads a Dreyeve frame given a sequence and the frame number
    
    Parameters
    ----------
    seq: int
        the sequence number.
    idx: int
        the frame number.

    Returns
    -------
    np.array
        the image.
    """

    seq_dir = join(dreyeve_root, 'DATA', '{:02d}'.format(seq), 'frames')

    img = io.imread(join(seq_dir, '{:06d}.jpg'.format(idx)))
    img = resize(img, output_shape=(1080 // 2, 1920 // 2), mode='constant')

    return img


def read_attention_map(seq, idx, which_map):
    """
    Reads an attentional map given the sequence, the frame number
    and the `which_map` parameter.

    Parameters
    ----------
    seq: int
        the sequence number.
    idx: int
        the frame number.
    which_map: str
        choose among [`groundtruth`, `prediction`, `central_baseline`]

    Returns
    -------
    np.array
        the attentional map.
    """
    if which_map == 'groundtruth':
        fix_dir = join(dreyeve_root, 'DATA', '{:02d}'.format(seq), 'saliency_fix')
        attention_map = io.imread(join(fix_dir, '{:06d}.png'.format(idx+1)))

    elif which_map == 'prediction':
        pred_dir = join(dreyeve_root, 'PREDICTIONS_2017', '{:02d}'.format(seq), 'dreyeveNet')
        attention_map = np.load(join(pred_dir, '{:06d}.npz'.format(idx)))['arr_0']

        attention_map /= np.max(attention_map)  # last activation is relu!
        attention_map *= 255
        attention_map = np.uint8(attention_map)
        attention_map = np.squeeze(attention_map)

    elif which_map == 'central_baseline':
        attention_map = io.imread(join(dreyeve_root, 'DATA', 'dreyeve_mean_train_gt_fix.png'))
    else:
        raise ValueError('Non valid value for which_map: {}'.format(which_map))

    attention_map = resize(attention_map, output_shape=shape, mode='constant')

    return attention_map


def get_alpha_from_attention_map(attention_map, sigma):
    """
    Transforms the attentional map into a blending factor to mix
    the original image and the blended one.
    The blended factor is evaluated as a negative exponential of the 
    euclidean distance transform of fixation points (should give a `conic flavor`).
    
    Parameters
    ----------
    attention_map: np.array
        the raw attention map.
    sigma: float
        the sigma of the rbf function.

    Returns
    -------
    alpha: np.array
        the blending factor.
    """
    # flatten attention map and get fixation idx
    attention_map_flat = np.reshape(attention_map, -1)
    fixation_idx_flat = attention_map_flat > 0.99  # todo how to recover fixations?

    # construct fixation dense map
    fixation_map_flat = np.ones_like(attention_map_flat)
    fixation_map_flat[fixation_idx_flat] = 0
    fixation_map = np.reshape(fixation_map_flat, shape)

    # get blending factor by means of fixation distance transform
    alpha = distance_transform_edt(fixation_map)
    alpha = np.exp(-alpha / sigma ** 2)
    alpha = np.expand_dims(alpha, axis=-1)

    return alpha


def write_video_specification(output_txt, video_name, which_map, seq, start, end):
    """
    Writes video information into the txt file (in append mode).
    
    Parameters
    ----------
    output_txt: str
        the output file to write to.
    video_name: str
        the name of the video file.
    which_map: str
        which map has been selected.
    seq: int
        the dreyeve sequence.
    start: int
        the start frame of the dreyeve sequence.
    end: int
        the stop frame of the dreyeve sequence.
    """

    with open(output_txt, 'a') as f:
        line = [video_name, which_map, seq, start, end]
        f.write(('{}\t'*len(line)).format(*line).rstrip())
        f.write('\n')


def main():
    """ Main script """

    # create output root if does not exist
    if not exists(output_root):
        os.makedirs(output_root)

    # sample a sequence
    seq = np.random.randint(38, 74 + 1)

    # sample a start point
    start = np.random.randint(15, 7498 - n_frames)

    # sample an attentional map
    which_map = np.random.choice(['groundtruth', 'prediction', 'central_baseline'])

    # sample a name
    video_name = str(uuid.uuid4()) + '.avi'
    video_path = join(output_root, video_name)

    # open videocapture
    ffmpeg_options = {
        '-b': '300000000'
    }
    writer = skvideo.io.FFmpegWriter(filename=video_path, outputdict=ffmpeg_options)

    for offset in tqdm(range(0, n_frames)):

        # read frame
        img = read_frame(seq=seq, idx=start+offset)

        # blur frame
        img_blur = gaussian(img, sigma=sigma_blur, multichannel=True)

        # read attention_map
        attention_map = read_attention_map(seq=seq, idx=start+offset, which_map=which_map)

        # get blending factor
        alpha = get_alpha_from_attention_map(attention_map, sigma=rbf_sigma)

        # blend
        blended = alpha * img + (1 - alpha) * img_blur
        # plt.imshow(blended)
        # plt.pause(0.02)

        writer.writeFrame(np.uint8(blended * 255))

    writer.close()

    # write video parameters on txt file
    write_video_specification(output_txt, video_name, which_map, seq, start, start + n_frames)


# entry point
if __name__ == '__main__':
    main()
