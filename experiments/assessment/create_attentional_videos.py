"""
This script is used to construct videos mimicking attentional behavior of
humans, deep model, central baseline. Such videos will be then used for
the quality assessment (Sec 5.4 of the paper).
"""
import numpy as np
from os.path import join, exists
import os

import cv2
import skimage.io as io
from skimage.transform import resize

import uuid
from tqdm import tqdm
import skvideo.io
from CtypesPermutohedralLattice import PermutohedralLattice

from scipy.ndimage.morphology import distance_transform_edt

from visualization.utils import blend_map

import matplotlib.pyplot as plt
plt.ion()

# parameters
dreyeve_root = '/majinbu/public/DREYEVE'
output_root = '/majinbu/public/DREYEVE/QUALITY_ASSESSMENT_VIDEOS_JET'
output_txt = join(output_root, 'videos.txt')
subsequences_txt = join(dreyeve_root, 'subsequences.txt')
n_frames = 1000
shape = (1080 // 2, 1920 // 2)

spatial_slopes = range(250, 950, 100)  # obsolete
color_slopes = range(250, 950, 100)  # obsolete


def get_driver_for_sequence(seq):
    """
    This function returns the driver id of a given sequence.
    
    Parameters
    ----------
    seq: int
        the sequence number
    Returns
    -------
    str
        the driver id
    """

    with open(join(dreyeve_root, 'dr(eye)ve_design.txt')) as f:
        dreyeve_design = np.array([f.rstrip().split('\t') for f in f.readlines()])

    row = np.where(dreyeve_design[:, 0] == '{:02d}'.format(seq))[0][0]
    driver_id = dreyeve_design[row, 4]

    return driver_id


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
    img = resize(img, output_shape=(1080 // 2, 1920 // 2), mode='constant', preserve_range=True)

    return np.uint8(img)


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

    elif which_map == 'central_baseline':
        attention_map = io.imread(join(dreyeve_root, 'DATA', 'dreyeve_mean_train_gt_fix.png'))
    else:
        raise ValueError('Non valid value for which_map: {}'.format(which_map))

    # attention_map /= np.max(attention_map)  # last activation is relu!
    # attention_map *= 255
    # attention_map = np.uint8(attention_map)
    attention_map = np.squeeze(attention_map)
    attention_map = np.float32(attention_map)
    attention_map /= np.sum(attention_map)

    attention_map = resize(attention_map, output_shape=shape, mode='constant')

    return attention_map


def blur_with_magic_permutho(img, attention_map, color_slope, spatial_slope):
    """
    Permutohedral blend. Obsolete?
    """

    # flatten attention map and get fixation idx
    attention_map_flat = np.reshape(attention_map, -1)
    fixation_idx_flat = np.argsort(attention_map_flat)[:-26:-1]

    # construct fixation dense map
    fixation_map_flat = np.ones_like(attention_map_flat)
    fixation_map_flat[fixation_idx_flat] = 0
    fixation_map = np.reshape(fixation_map_flat, shape)

    # get blending factor by means of fixation distance transform
    radial_sigma = distance_transform_edt(fixation_map)
    radial_sigma = np.expand_dims(radial_sigma, axis=-1)
    radial_sigma /= radial_sigma.max()

    # get sigma for color features
    radial_sigma_color = radial_sigma * (color_slope - 1)
    radial_sigma_color += 1
    radial_sigma_color = np.float32(radial_sigma_color)

    # get sigma for spatial features
    radial_sigma_spatial = radial_sigma * (spatial_slope - 1)
    radial_sigma_spatial += 1
    radial_sigma_spatial = np.float32(radial_sigma_spatial)

    # use lattice! D=
    h, w = attention_map.shape
    x_range = range(0, w)
    y_range = range(0, h)

    cols, rows = np.meshgrid(x_range, y_range)
    grid = np.stack((rows, cols), axis=-1)
    grid = np.float32(grid)

    color_features = (img * 255) / radial_sigma_color
    spatial_features = grid / radial_sigma_spatial
    features = np.concatenate((color_features, spatial_features), axis=-1)

    lattice = PermutohedralLattice(features)
    blurred = lattice.compute(img, normalize=True)

    return blurred, features


def write_video_specification(filename, video_name, driver_id, which_map, seq,
                              start, end, is_acting):
    """
    Writes video information into the txt file (in append mode).
    
    Parameters
    ----------
    filename: str
        the output file to write to.
    video_name: str
        the name of the video file.
    driver_id: str
        the id of the driver.
    which_map: str
        which map has been selected.
    seq: int
        the dreyeve sequence.
    start: int
        the start frame of the dreyeve sequence.
    end: int
        the stop frame of the dreyeve sequence.
    is_acting: bool
        whether the sequence contains acting subsequences or not.
    """

    with open(filename, 'a') as f:
        line = [video_name, driver_id, which_map, seq, start, end, is_acting]
        f.write(('{}\t'*len(line)).format(*line).rstrip())
        f.write('\n')


def get_random_clip():
    """
    This function returns a random clip.
    
    Returns
    -------
    tuple
        a tuple like (seq, start_frame, contains_acting).
    """

    with open(subsequences_txt, mode='r') as f:
        subsequences = np.array([l.rstrip().split('\t') for l in f.readlines()])

    acting_subseqs = subsequences[subsequences[:, 3] == 'k', :3]
    acting_subseqs = np.int32(acting_subseqs)

    sequences = range(38, 74 + 1)
    seq_probs = np.array([np.shape(acting_subseqs[acting_subseqs[:, 0] == s])[0] for s in sequences], dtype=np.float32)
    seq_probs /= np.sum(seq_probs)

    contains_acting = np.random.choice(['acting', 'non_acting'])

    while True:
        if contains_acting == 'acting':
            seq = np.random.choice(sequences, p=seq_probs)

            acting_subseqs = acting_subseqs[acting_subseqs[:, 0] == seq]

            start_probs = np.zeros(shape=7500, dtype=np.float32)
            for _, start, stop in acting_subseqs:
                start = max(0, start - n_frames)
                stop = max(0, stop - n_frames)

                start_probs[start:stop] += 1

            start_probs[-n_frames:] = 0
            start_probs[0] = 0
            start_probs /= np.sum(start_probs)

            start = np.random.choice(range(0, 7500), p=start_probs)

        else:

            seq = np.random.choice(sequences)

            acting_subseqs = acting_subseqs[acting_subseqs[:, 0] == seq]

            start_probs = np.ones(shape=7500, dtype=np.float32)
            for _, start, stop in acting_subseqs:
                start = max(0, start - n_frames)

                start_probs[start:stop] = 0

            start_probs[-n_frames:] = 0
            start_probs[0] = 0
            start_probs /= np.sum(start_probs)

            start = np.random.choice(range(0, 7500), p=start_probs)

        if start != 0:  # exit
            break

    return seq, start, contains_acting


def main():
    """ Main script """

    # create output root if does not exist
    if not exists(output_root):
        os.makedirs(output_root)

    # sample a sequence and a start frame
    seq, start, is_acting = get_random_clip()

    # sample slopes of peripheral decay
    # color_slope = np.random.choice(color_slopes)
    # spatial_slope = np.random.choice(spatial_slopes)

    # get driver for sequence
    driver_id = get_driver_for_sequence(seq)

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

    # f, axs = plt.subplots(1, 7)
    for offset in tqdm(range(0, n_frames)):

        # read frame
        img = read_frame(seq=seq, idx=start+offset)

        # read attention_map
        attention_map = read_attention_map(seq=seq, idx=start+offset, which_map=which_map)

        # permutohedral radial blend
        # blended, features = blur_with_magic_permutho(img, attention_map, color_slope, spatial_slope)
        blended = blend_map(img, attention_map, factor=0.5)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        # blend
        # blended = alpha * img + (1 - alpha) * img_blur
        # for i, ax in zip([img, blended] + [f for f in features.transpose(2, 0, 1)], axs):
        #     ax.imshow(i)
        #
        # plt.pause(0.02)

        writer.writeFrame(np.uint8(blended))

    writer.close()

    # write video parameters on txt file
    write_video_specification(output_txt, video_name, driver_id, which_map, seq,
                              start, start + n_frames, is_acting)


# entry point
if __name__ == '__main__':
    for _ in range(0, 10000):
        main()
