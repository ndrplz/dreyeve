import numpy as np
import cv2

from random import choice
from os.path import join
from time import time

from config import dreyeve_dir, frame_size_before_crop, simo_mode, total_frames_each_run, force_sample_steering
from config import dreyeve_train_seq, dreyeve_test_seq
from config import train_frame_range, val_frame_range, test_frame_range
from config import crop_type
from utils import palette, read_lines_from_file

from computer_vision_utils.io_helper import read_image, normalize
from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.tensor_manipulation import resize_tensor, crop_tensor


def sample_signature(sequences, allowed_frames, image_size, allow_mirror):
    """
    Function to create a unique batch signature for the Dreyeve dataset.

    :param sequences: sequences to sample from.
    :param allowed_frames: range of allowed frames to sample the sequence start from.
    :param image_size: in the form (h,w). Needed to crop randomly.
    :param allow_mirror: whether or not to enable random mirroring.
    :return: a tuple like (num_run, start, hc1, hc2, wc1, wc2, do_mirror).
    """
    h, w = image_size
    h_c = h // 4
    w_c = w // 4
    h_before_crop, w_before_crop = frame_size_before_crop

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
    p *= mask
    p /= np.sum(p)
    start = np.random.choice(range(0, total_frames_each_run), p=p)

    # get random crop
    if crop_type == 'central':
        hc1 = h_before_crop // 4
        wc1 = w_before_crop // 4
    elif crop_type == 'random':
        hc1 = np.random.randint(0, h_before_crop - h_c)
        wc1 = np.random.randint(0, w_before_crop - w_c)
    else:
        raise ValueError
    
    hc2 = hc1 + h_c
    wc2 = wc1 + w_c

    do_mirror = choice([True, False]) if allow_mirror else False

    return tuple((num_run, start, hc1, hc2, wc1, wc2, do_mirror))


def load_batch_data(signatures, nb_frames, image_size, batch_type):
    """
    Function to load a data batch. This is common for `image`, `optical_flow` and `semseg`.

    :param signatures: sample signatures, previously evaluated. List of tuples like
                    (num_run, start, hc1, hc2, wc1, wc2). The list is batchsize signatures long.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param batch_type: choose among [`image`, `optical_flow`, `semseg`].
    :return: a tuple holding the fullframe, the small and the cropped batch.
    """
    assert batch_type in ['image', 'optical_flow', 'semseg'], 'Unknown batch type: {}'.format(batch_type)

    batchsize = len(signatures)
    h, w = image_size
    h_s = h_c = h // 4
    w_s = w_c = w // 4

    if batch_type == 'image':
        B_ff = np.zeros(shape=(batchsize, 3, 1, h, w), dtype=np.float32)
        B_s = np.zeros(shape=(batchsize, 3, nb_frames, h_s, w_s), dtype=np.float32)
        B_c = np.zeros(shape=(batchsize, 3, nb_frames, h_c, w_c), dtype=np.float32)
        subdir = 'frames'
        # mean frame image, to subtract mean
        mean_image = read_image(join(dreyeve_dir, 'dreyeve_mean_frame.png'), channels_first=True, resize_dim=image_size)
    elif batch_type == 'optical_flow':
        B_ff = np.zeros(shape=(batchsize, 3, 1, h, w), dtype=np.float32)
        B_s = np.zeros(shape=(batchsize, 3, nb_frames, h_s, w_s), dtype=np.float32)
        B_c = np.zeros(shape=(batchsize, 3, nb_frames, h_c, w_c), dtype=np.float32)
        subdir = 'optical_flow'
    elif batch_type == 'semseg':
        B_ff = np.zeros(shape=(batchsize, 19, 1, h, w), dtype=np.float32)
        B_s = np.zeros(shape=(batchsize, 19, nb_frames, h_s, w_s), dtype=np.float32)
        B_c = np.zeros(shape=(batchsize, 19, nb_frames, h_c, w_c), dtype=np.float32)
        subdir = 'semseg'

    for b in range(batchsize):
        # retrieve the signature
        num_run, start, hc1, hc2, wc1, wc2, do_mirror = signatures[b]

        data_dir = join(dreyeve_dir, '{:02d}'.format(num_run), subdir)

        for offset in range(0, nb_frames):
            if batch_type == 'image':
                x = read_image(join(data_dir, '{:06d}.jpg'.format(start + offset)),
                               channels_first=True, resize_dim=image_size) - mean_image

                # resize to (256, 256) before cropping
                x_before_crop = resize_tensor(x, new_size=frame_size_before_crop)

                B_s[b, :, offset, :, :] = resize_tensor(x, new_size=(h_s, w_s))
                B_c[b, :, offset, :, :] = crop_tensor(x_before_crop, indexes=(hc1, hc2, wc1, wc2))
            elif batch_type == 'optical_flow':
                x = read_image(join(data_dir, '{:06d}.png'.format(start + offset + 1)),
                               channels_first=True, resize_dim=image_size)
                x -= np.mean(x, axis=(1, 2), keepdims=True)  # remove mean

                # resize to (256, 256) before cropping
                x_before_crop = resize_tensor(x, new_size=frame_size_before_crop)

                B_s[b, :, offset, :, :] = resize_tensor(x, new_size=(h_s, w_s))
                B_c[b, :, offset, :, :] = crop_tensor(x_before_crop, indexes=(hc1, hc2, wc1, wc2))
            elif batch_type == 'semseg':
                x = resize_tensor(np.load(join(data_dir, '{:06d}.npz'.format(start + offset)))['arr_0'][0],
                                  new_size=image_size)
                # resize to (256, 256) before cropping
                x_before_crop = resize_tensor(x, new_size=frame_size_before_crop)

                B_s[b, :, offset, :, :] = resize_tensor(x, new_size=(h_s, w_s))
                B_c[b, :, offset, :, :] = crop_tensor(x_before_crop, indexes=(hc1, hc2, wc1, wc2))
        B_ff[b, :, 0, :, :] = x

        if do_mirror:
            B_ff = B_ff[:, :, :, :, ::-1]
            B_s = B_s[:, :, :, :, ::-1]
            B_c = B_c[:, :, :, :, ::-1]

    return [B_ff, B_s, B_c]


def load_saliency_data(signatures, nb_frames, image_size, gt_type):
    """
    Function to load a saliency batch.

    :param signatures: sample signatures, previously evaluated. List of tuples like
                    (num_run, start, hc1, hc2, wc1, wc2). The list is batchsize signatures long.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple holding the fullframe and the cropped saliency.
    """

    batchsize = len(signatures)
    h, w = image_size
    h_c = h // 4
    w_c = w // 4

    Y = np.zeros(shape=(batchsize, 1, h, w), dtype=np.float32)
    Y_c = np.zeros(shape=(batchsize, 1, h_c, w_c), dtype=np.float32)

    for b in range(0, batchsize):
        # retrieve the signature
        num_run, start, hc1, hc2, wc1, wc2, do_mirror = signatures[b]

        y_dir = join(dreyeve_dir, '{:02d}'.format(num_run), 'saliency' if gt_type == 'sal' else 'saliency_fix')
        # saliency
        y = read_image(join(y_dir, '{:06d}.png'.format(start + nb_frames - 1)),
                       channels_first=True, color=False, resize_dim=image_size)

        # resize to (256, 256) before cropping
        y_before_crop = resize_tensor(np.expand_dims(y, axis=0), new_size=frame_size_before_crop)

        Y[b, 0, :, :] = y
        Y_c[b, 0, :, :] = crop_tensor(y_before_crop, indexes=(hc1, hc2, wc1, wc2))[0]

        if do_mirror:
            Y = Y[:, :, :, ::-1]
            Y_c = Y_c[:, :, :, ::-1]

    return [Y, Y_c]


def load_saliency_data_simo(signatures, nb_frames, image_size):
    """
    Crop -> NSS with fixations
    Fine -> KLD with saliency maps and IG with fixations (baseline is saliency maps)

    :param signatures: sample signatures, previously evaluated. List of tuples like
                    (num_run, start, hc1, hc2, wc1, wc2). The list is batchsize signatures long.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :return: a tuple holding the fullframe and the cropped saliency.
    """

    batchsize = len(signatures)
    h, w = image_size
    h_c = h // 4
    w_c = w // 4

    Y = np.zeros(shape=(batchsize, 2, h, w), dtype=np.float32)
    Y_c = np.zeros(shape=(batchsize, 1, h_c, w_c), dtype=np.float32)

    for b in range(0, batchsize):
        # retrieve the signature
        num_run, start, hc1, hc2, wc1, wc2, do_mirror = signatures[b]

        y_dir_sal = join(dreyeve_dir, '{:02d}'.format(num_run), 'saliency')
        y_dir_fix = join(dreyeve_dir, '{:02d}'.format(num_run), 'saliency_fix')
        # saliency
        y_sal = read_image(join(y_dir_sal, '{:06d}.png'.format(start + nb_frames - 1)),
                           channels_first=True, color=False, resize_dim=image_size) / 255
        y_fix = read_image(join(y_dir_fix, '{:06d}.png'.format(start + nb_frames - 1)),
                           channels_first=True, color=False, resize_dim=image_size) / 255

        # resize to (256, 256) before cropping
        y_before_crop = resize_tensor(np.expand_dims(y_fix, axis=0), new_size=frame_size_before_crop)

        Y[b, 0, :, :] = y_sal
        Y[b, 1, :, :] = y_fix
        Y_c[b, 0, :, :] = crop_tensor(y_before_crop, indexes=(hc1, hc2, wc1, wc2))[0]

        if do_mirror:
            Y = Y[:, :, :, ::-1]
            Y_c = Y_c[:, :, :, ::-1]

    return [Y, Y_c]


def dreyeve_I_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to load a Dreyeve batch of only images.

    :param batchsize: batchsize.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param mode: choose among [`train`, `val`, `test`].
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: an image batch and the relative saliency in the form (X,Y).
    """
    assert mode in ['train', 'val', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['sal', 'fix'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)
    if mode == 'train':
        sequences = dreyeve_train_seq
        allowed_frames = train_frame_range
        allow_mirror = True
    elif mode == 'val':
        sequences = dreyeve_train_seq
        allowed_frames = val_frame_range
        allow_mirror = False
    elif mode == 'test':
        sequences = dreyeve_test_seq
        allowed_frames = test_frame_range
        allow_mirror = False

    # generate batch signatures
    signatures = []
    for b in range(0, batchsize):
        signatures.append(sample_signature(sequences=sequences, allowed_frames=allowed_frames,
                                           image_size=image_size, allow_mirror=allow_mirror))

    # get an image batch
    I = load_batch_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, batch_type='image')
    if simo_mode:
        Y = load_saliency_data_simo(signatures=signatures, nb_frames=nb_frames, image_size=image_size)
    else:
        Y = load_saliency_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, gt_type=gt_type)
    return I, Y


def dreyeve_OF_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to load a Dreyeve batch of only optical flow data.

    :param batchsize: batchsize.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param mode: choose among [`train`, `val`, `test`].
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: an optical flow batch and the relative saliency in the form (X,Y).
    """
    assert mode in ['train', 'val', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['sal', 'fix'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)
    if mode == 'train':
        sequences = dreyeve_train_seq
        allowed_frames = train_frame_range
        allow_mirror = True
    elif mode == 'val':
        sequences = dreyeve_train_seq
        allowed_frames = val_frame_range
        allow_mirror = False
    elif mode == 'test':
        sequences = dreyeve_test_seq
        allowed_frames = test_frame_range
        allow_mirror = False

    # generate batch signatures
    signatures = []
    for b in range(0, batchsize):
        signatures.append(sample_signature(sequences=sequences, allowed_frames=allowed_frames,
                                           image_size=image_size, allow_mirror=allow_mirror))

    # get an optical flow batch
    OF = load_batch_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, batch_type='optical_flow')
    if simo_mode:
        Y = load_saliency_data_simo(signatures=signatures, nb_frames=nb_frames, image_size=image_size)
    else:
        Y = load_saliency_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, gt_type=gt_type)
    return OF, Y


def dreyeve_SEG_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to load a Dreyeve batch of only segmentation data.

    :param batchsize: batchsize.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param mode: choose among [`train`, `val`, `test`].
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a segmentation batch and the relative saliency in the form (X,Y).
    """
    assert mode in ['train', 'val', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['sal', 'fix'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)
    if mode == 'train':
        sequences = dreyeve_train_seq
        allowed_frames = train_frame_range
        allow_mirror = True
    elif mode == 'val':
        sequences = dreyeve_train_seq
        allowed_frames = val_frame_range
        allow_mirror = False
    elif mode == 'test':
        sequences = dreyeve_test_seq
        allowed_frames = test_frame_range
        allow_mirror = False

    # generate batch signatures
    signatures = []
    for b in range(0, batchsize):
        signatures.append(sample_signature(sequences=sequences, allowed_frames=allowed_frames,
                                           image_size=image_size, allow_mirror=allow_mirror))

    # get an segmentation batch
    SEG = load_batch_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, batch_type='semseg')
    if simo_mode:
        Y = load_saliency_data_simo(signatures=signatures, nb_frames=nb_frames, image_size=image_size)
    else:
        Y = load_saliency_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, gt_type=gt_type)
    return SEG, Y


def dreyeve_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to load a batch of the dreyeve dataset, with image, optical flow and segmentation.

    :param batchsize: batchsize.
    :param nb_frames: number of frames for each batch.
    :param image_size: dimension of tensors, must satisfy (h,w) % 4 = (0,0).
    :param mode: choose among [`train`, `val`, `test`].
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple like: [X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c], [Y, Y_c].
    """
    assert mode in ['train', 'val', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['sal', 'fix'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)
    if mode == 'train':
        sequences = dreyeve_train_seq
        allowed_frames = train_frame_range
        allow_mirror = True
    elif mode == 'val':
        sequences = dreyeve_train_seq
        allowed_frames = val_frame_range
        allow_mirror = False
    elif mode == 'test':
        sequences = dreyeve_test_seq
        allowed_frames = test_frame_range
        allow_mirror = False

    # generate batch signatures
    signatures = []
    for b in range(0, batchsize):
        signatures.append(sample_signature(sequences=sequences, allowed_frames=allowed_frames,
                                           image_size=image_size, allow_mirror=allow_mirror))

    # get all batches
    I = load_batch_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, batch_type='image')
    OF = load_batch_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, batch_type='optical_flow')
    SEG = load_batch_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, batch_type='semseg')
    if simo_mode:
        Y = load_saliency_data_simo(signatures=signatures, nb_frames=nb_frames, image_size=image_size)
    else:
        Y = load_saliency_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, gt_type=gt_type)
    return I + OF + SEG, Y  # list "+" concatenates


def generate_dreyeve_I_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to generate a batch from the dreyeve dataset. The batch will only contain images.

    :param batchsize: batchsize.
    :param nb_frames: number of frames for each batch.
    :param image_size: dimension of tensors.
    :param mode: `train` or `test`.
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple like: ([X, X_s, X_c], [Y, Y_c]).
    """
    while True:
        yield dreyeve_I_batch(batchsize=batchsize, nb_frames=nb_frames,
                              image_size=image_size, mode=mode, gt_type=gt_type)


def generate_dreyeve_OF_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to generate a batch from the dreyeve dataset. The batch will only contain optical flow.

    :param batchsize: batchsize.
    :param nb_frames: number of frames for each batch.
    :param image_size: dimension of tensors.
    :param mode: `train` or `test`.
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple like: ([OF, OF_s, OF_c], [Y, Y_c]).
    """
    while True:
        yield dreyeve_OF_batch(batchsize=batchsize, nb_frames=nb_frames,
                               image_size=image_size, mode=mode, gt_type=gt_type)


def generate_dreyeve_SEG_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to generate a batch from the dreyeve dataset. The batch will only contain segmentation data.

    :param batchsize: batchsize.
    :param nb_frames: number of frames for each batch.
    :param image_size: dimension of tensors.
    :param mode: `train` or `test`.
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple like: ([SEG, SEG_s, SEG_c], [Y, Y_c]).
    """
    while True:
        yield dreyeve_SEG_batch(batchsize=batchsize, nb_frames=nb_frames,
                                image_size=image_size, mode=mode, gt_type=gt_type)


def generate_dreyeve_batch(batchsize, nb_frames, image_size, mode, gt_type='fix'):
    """
    Function to generate a batch from the dreyeve dataset.

    :param batchsize: batchsize.
    :param nb_frames: number of frames for each batch.
    :param image_size: dimension of tensors.
    :param mode: `train` or `test`.
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple like: ([X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c], [Y, Y_c]).
    """
    while True:
        yield dreyeve_batch(batchsize=batchsize, nb_frames=nb_frames,
                            image_size=image_size, mode=mode, gt_type=gt_type)


def visualize_batch(X, Y):
    """
    Helper function to visualize a batch.

    :param X: input data: [X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c].
    :param Y: saliency data like [Y, Y_c].
    """
    batchsize, _, frames_per_batch, h, w = X[0].shape
    batchsize, _, frames_per_batch, h_s, w_s = X[1].shape
    batchsize, _, frames_per_batch, h_c, w_c = X[2].shape

    X, X_s, X_c, OF, OF_s, OF_c, SEG, SEG_s, SEG_c = X
    Y, Y_c = Y
    for b in range(0, batchsize):
        for f in range(0, frames_per_batch):
            # FULL FRAME SECTION -----
            x = X[b, :, 0, :, :].transpose(1, 2, 0)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

            of = OF[b, :, 0, :, :].transpose(1, 2, 0)
            of = cv2.cvtColor(of, cv2.COLOR_RGB2BGR)

            # seg is different, we have to turn into colors
            seg = SEG[b, :, 0, :, :]
            seg = palette[np.argmax(seg, axis=0).ravel()].reshape(h, w, 3)
            seg = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)

            # we have to turn y to 3 channels 255 for stitching
            y = Y[b, 0, :, :]
            y = (np.tile(y, (3, 1, 1))).transpose(1, 2, 0)

            # stitch and visualize
            stitch_ff = stitch_together([normalize(x), of, seg, normalize(y)], layout=(2, 2), resize_dim=(540, 960))

            # CROPPED FRAME SECTION -----
            x_c = X_c[b, :, f, :, :].transpose(1, 2, 0)
            x_c = cv2.cvtColor(x_c, cv2.COLOR_RGB2BGR)

            of_c = OF_c[b, :, f, :, :].transpose(1, 2, 0)
            of_c = cv2.cvtColor(of_c, cv2.COLOR_RGB2BGR)

            # seg is different, we have to turn into colors
            seg_c = SEG_c[b, :, f, :, :]
            seg_c = palette[np.argmax(seg_c, axis=0).ravel()].reshape(h_c, w_c, 3)
            seg_c = cv2.cvtColor(seg_c, cv2.COLOR_RGB2BGR)

            # we have to turn y to 3 channels 255 for stitching
            y_c = Y_c[b, 0, :, :]
            y_c = (np.tile(y_c, (3, 1, 1))).transpose(1, 2, 0)

            # stitch and visualize
            stitch_c = stitch_together([normalize(x_c), of_c, seg_c, normalize(y_c)], layout=(2, 2), resize_dim=(540, 960))

            # SMALL FRAME SECTION -----
            x_s = X_s[b, :, f, :, :].transpose(1, 2, 0)
            x_s = cv2.cvtColor(x_s, cv2.COLOR_RGB2BGR)

            of_s = OF_s[b, :, f, :, :].transpose(1, 2, 0)
            of_s = cv2.cvtColor(of_s, cv2.COLOR_RGB2BGR)

            # seg is different, we have to turn into colors
            seg_s = SEG_s[b, :, f, :, :]
            seg_s = palette[np.argmax(seg_s, axis=0).ravel()].reshape(h_s, w_s, 3)
            seg_s = cv2.cvtColor(seg_s, cv2.COLOR_RGB2BGR)

            # we have to turn y to 3 channels 255 for stitching
            # also, we resize it to small (just for visualize it)
            y_s = cv2.resize(Y[b, 0, :, :], dsize=(h_s, w_s)[::-1])
            y_s = (np.tile(y_s, (3, 1, 1))).transpose(1, 2, 0)

            # stitch and visualize
            stitch_s = stitch_together([normalize(x_s), of_s, seg_s, normalize(y_s)],
                                       layout=(2, 2), resize_dim=(540, 960))

            # stitch the stitchs D=
            final_stitch = stitch_together([stitch_ff, stitch_s, stitch_c],
                                           layout=(1, 3), resize_dim=(810, 1440))
            cv2.imshow('Batch_viewer', final_stitch.astype(np.uint8))
            cv2.waitKey()


def test_load_batch():
    """
    Helper function, to load and visualize a dreyeve batch.
    """
    t = time()
    X, Y = dreyeve_batch(batchsize=8, nb_frames=16, image_size=(448, 800), mode='train', gt_type='fix')
    elapsed = time() - t

    print 'Batch loaded in {} seconds.'.format(elapsed)
    print 'I shape:{}'.format(X[0].shape)
    print 'I_s shape:{}'.format(X[1].shape)
    print 'I_c shape:{}'.format(X[2].shape)
    print 'OF shape:{}'.format(X[3].shape)
    print 'OF_s shape:{}'.format(X[4].shape)
    print 'OF_c shape:{}'.format(X[5].shape)
    print 'SEG shape:{}'.format(X[6].shape)
    print 'SEG_s shape:{}'.format(X[7].shape)
    print 'SEG_c shape:{}'.format(X[8].shape)
    print 'Y shape:{}'.format(Y[0].shape)
    print 'Y_c shape:{}'.format(Y[1].shape)

    visualize_batch(X, Y)


# tester function
if __name__ == '__main__':
    test_load_batch()
