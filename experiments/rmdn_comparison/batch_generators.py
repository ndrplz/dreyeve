import numpy as np

from computer_vision_utils.io_helper import read_image

from os.path import join

from config import DREYEVE_ROOT
from config import dreyeve_train_seq, dreyeve_test_seq
from config import train_frame_range, val_frame_range, test_frame_range
from config import T, h, w, encoding_dim


def RMDN_batch(batchsize, mode):
    """
    Function to provide a batch for RMDN training.

    :param batchsize: batchsize.
    :param mode: choose among [`train`, `val`, `test`].
    :return: a batch like sequence numbers, frame ids, X, Y
    """
    assert mode in ['train', 'val', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)

    if mode == 'train':
        sequences = dreyeve_train_seq
        allowed_frames = train_frame_range
    elif mode == 'val':
        sequences = dreyeve_train_seq
        allowed_frames = val_frame_range
    elif mode == 'test':
        sequences = dreyeve_test_seq
        allowed_frames = test_frame_range

    seqs = np.zeros(shape=(batchsize, T), dtype=np.uint32)
    frs = np.zeros(shape=(batchsize, T), dtype=np.uint32)
    X = np.zeros(shape=(batchsize, T, encoding_dim), dtype=np.float32)
    Y = np.zeros(shape=(batchsize, T, h, w))
    for b in xrange(0, batchsize):
        # sample a sequence
        seq = np.random.choice(sequences)
        seq_enc_dir = join(DREYEVE_ROOT, '{:02d}'.format(seq), 'c3d_encodings')
        seq_gt_dir = join(DREYEVE_ROOT, '{:02d}'.format(seq), 'saliency_fix')

        # choose a random sample
        fr = np.random.choice(allowed_frames)
        for offset in range(0, T):
            c3d_encoding = np.load(join(seq_enc_dir, '{:06d}.npz'.format(fr+offset)))['arr_0']
            gt = read_image(join(seq_gt_dir, '{:06d}.png'.format(fr+offset+1)), channels_first=False, color=False,
                            resize_dim=(h, w))

            seqs[b, offset] = seq
            frs[b, offset] = fr+offset
            X[b, offset] = c3d_encoding
            Y[b, offset] = gt

    return seqs, frs, X, Y


def generate_RMDN_batch(batchsize, mode):
    """
    Function that yields batches for RMDN training continuously.

    :param batchsize: batchsize.
    :param mode: choose among [`train`, `val`, `test`].
    :return: a batch like X, Y
    """
    while True:
        _, _, X, Y = RMDN_batch(batchsize, mode)
        yield X, Y


# helper function to test batch loading.
if __name__ == '__main__':
    _, _, X, Y = RMDN_batch(batchsize=8, mode='train')

    print X.shape
    print Y.shape
