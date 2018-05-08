import argparse
import numpy as np
from os.path import join
from train.models import DreyeveNet
from train.config import dreyeve_test_seq
from train.config import frames_per_seq
from train.config import h
from train.config import w
from tqdm import tqdm
from metrics.metrics import kld_numeric
from computer_vision_utils.io_helper import read_image
from computer_vision_utils.tensor_manipulation import resize_tensor


def translate_tensor(x, pixels):

    if pixels < 0:
        side = 'left'
        pixels = -pixels
    elif pixels > 0:
        side = 'right'
    else:
        return x

    w = x.shape[-1]

    pad = x[..., (w - pixels):] if side == 'left' else x[..., :pixels]
    pad = pad[..., ::-1]

    if side == 'left':
        xt = np.roll(x, -pixels, axis=-1)
        xt[..., (w-pixels):] = pad
    else:
        xt = np.roll(x, pixels, axis=-1)
        xt[..., :pixels] = pad

    return xt


def translate_batch(batch, pixels):

    X, Y = [[np.copy(a) for a in b] for b in batch]
    for i, tensor in enumerate(X):
        tensor_shift = (tensor.shape[-1] * pixels) // 1920
        X[i] = translate_tensor(tensor, pixels=tensor_shift)
    for i, tensor in enumerate(Y):
        tensor_shift = (tensor.shape[-1] * pixels) // 1920
        Y[i] = translate_tensor(tensor, pixels=tensor_shift)
    return X, Y


def load_dreyeve_sample(sequence_dir, stop, mean_dreyeve_image):
    """
    Function to load a dreyeve_sample.

    :param sequence_dir: string, sequence directory (e.g. 'Z:/DATA/04/').
    :param stop: int, sample to load in (15, 7499). N.B. this is the sample where prediction occurs!
    :param mean_dreyeve_image: mean dreyeve image, subtracted to each frame.
    :param frames_per_seq: number of temporal frames for each sample
    :param h: h
    :param w: w
    :return: a dreyeve_sample like I, OF, SEG
    """

    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    I_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    I_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')
    OF_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    OF_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    OF_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')
    SEG_ff = np.zeros(shape=(1, 19, 1, h, w), dtype='float32')
    SEG_s = np.zeros(shape=(1, 19, frames_per_seq, h_s, w_s), dtype='float32')
    SEG_c = np.zeros(shape=(1, 19, frames_per_seq, h_c, w_c), dtype='float32')

    Y_sal = np.zeros(shape=(1, 1, h, w), dtype='float32')
    Y_fix = np.zeros(shape=(1, 1, h, w), dtype='float32')

    for fr in xrange(0, frames_per_seq):
        offset = stop - frames_per_seq + 1 + fr   # tricky

        # read image
        x = read_image(join(sequence_dir, 'frames', '{:06d}.jpg'.format(offset)),
                       channels_first=True, resize_dim=(h, w)) - mean_dreyeve_image
        I_s[0, :, fr, :, :] = resize_tensor(x, new_size=(h_s, w_s))

        # read of
        of = read_image(join(sequence_dir, 'optical_flow', '{:06d}.png'.format(offset + 1)),
                        channels_first=True, resize_dim=(h, w))
        of -= np.mean(of, axis=(1, 2), keepdims=True)  # remove mean
        OF_s[0, :, fr, :, :] = resize_tensor(of, new_size=(h_s, w_s))

        # read semseg
        seg = resize_tensor(np.load(join(sequence_dir, 'semseg', '{:06d}.npz'.format(offset)))['arr_0'][0],
                            new_size=(h, w))
        SEG_s[0, :, fr, :, :] = resize_tensor(seg, new_size=(h_s, w_s))

    I_ff[0, :, 0, :, :] = x
    OF_ff[0, :, 0, :, :] = of
    SEG_ff[0, :, 0, :, :] = seg

    Y_sal[0, 0] = read_image(join(sequence_dir, 'saliency', '{:06d}.png'.format(stop)), channels_first=False,
                             color=False, resize_dim=(h, w))
    Y_fix[0, 0] = read_image(join(sequence_dir, 'saliency_fix', '{:06d}.png'.format(stop)), channels_first=False,
                             color=False, resize_dim=(h, w))

    return [I_ff, I_s, I_c, OF_ff, OF_s, OF_c, SEG_ff, SEG_s, SEG_c], [Y_sal, Y_fix]


class DataLoader:

    def __init__(self, dreyeve_root):

        self.dreyeve_root = dreyeve_root
        self.dreyeve_data_root = join(dreyeve_root, 'DATA')
        self.subseq_file = join(dreyeve_root, 'subsequences.txt')

        # load subsequences
        self.subseqs = np.loadtxt(self.subseq_file, dtype=str)

        # filter attentive
        self.subseqs = self.subseqs[self.subseqs[:, -1] == 'k']

        # cast to int
        self.subseqs = np.int32(self.subseqs[:, :-1])

        # filter test sequences
        self.subseqs = np.array([seq for seq in self.subseqs if seq[0] in dreyeve_test_seq])

        # filter too short sequences
        self.subseqs = np.array([seq for seq in self.subseqs if seq[2] - seq[1] >= frames_per_seq])

        self.len = len(self.subseqs)
        self.counter = 0

        # load mean dreyeve image
        self.mean_dreyeve_image = read_image(join(self.dreyeve_data_root, 'dreyeve_mean_frame.png'),
                                             channels_first=True, resize_dim=(h, w))

    def __len__(self):
        return self.len

    def get_sample(self):

        # compute center of this subsequence
        seq, start, stop = self.subseqs[self.counter]
        # start = (start + stop) / 2 - frames_per_seq / 2
        start = np.random.randint(0, 7500 - frames_per_seq)
        stop = start + frames_per_seq

        # compute sequence dir
        sequence_dir = join(self.dreyeve_data_root, '{:02d}'.format(seq))
        batch = load_dreyeve_sample(sequence_dir, stop, self.mean_dreyeve_image)

        self.counter += 1

        return batch


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str)
    args = parser.parse_args()

    assert args.checkpoint_file is not None, 'Please provide a checkpoint model to load.'

    # get the models
    dreyevenet_model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    dreyevenet_model.compile(optimizer='adam', loss='kld')  # do we need this?
    dreyevenet_model.load_weights(args.checkpoint_file)  # load weights

    dreyeve_root = '/majinbu/public/DREYEVE'
    shifts = np.arange(-800, 801, step=200)

    # get data_loader
    loader = DataLoader(dreyeve_root)

    # set up array for kld results
    kld_results = np.zeros(shape=(len(loader), len(shifts)))

    for clip_idx in tqdm(range(0, len(loader))):

        batch = loader.get_sample()

        # compute shifted versions
        X_list = []
        GT_list = []
        for s in shifts:
            X, GT = translate_batch(batch, pixels=s)
            X_list.append(X)
            GT_list.append(GT)

        X_batch = [np.concatenate(l) for l in zip(*X_list)]
        GT_batch = [np.concatenate(l) for l in zip(*GT_list)][1]

        P_batch = dreyevenet_model.predict(X_batch)[0]

        for shift_idx, (p, gt) in enumerate(zip(P_batch, GT_batch)):
            kld_results[clip_idx, shift_idx] = kld_numeric(gt, p)

    np.savetxt(args.checkpoint_file + '.txt',
               X=np.concatenate(
                   (
                       np.mean(kld_results, axis=0, keepdims=True),
                       np.std(kld_results, axis=0, keepdims=True)
                   ),
                   axis=0
               ))
