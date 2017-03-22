import numpy as np

import argparse

import os
from tqdm import tqdm
from os.path import join

from computer_vision_utils.io_helper import read_image

from metrics import kld_numeric, cc_numeric


class MetricSaver:
    """
    This class handles metric computation given predictions and groundtruth.

    Params:
    pred_dir: the prediction directory
    seq: the number of the sequence
    """
    def __init__(self, pred_dir, seq):
        self.metrics_dir = join(pred_dir, '{:02d}'.format(seq), 'metrics')
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

        # open files and put headers in it
        self.kld_file = open(join(self.metrics_dir, 'kld.txt'), mode='w')
        self.kld_file.write('FRAME_NUMBER,'
                            'KLD_DREYEVE_WRT_SAL,'
                            'KLD_IMAGE_WRT_SAL,'
                            'KLD_FLOW_WRT_SAL,'
                            'KLD_SEG_WRT_SAL,'
                            'KLD_DREYEVE_WRT_FIX,'
                            'KLD_IMAGE_WRT_FIX,'
                            'KLD_FLOW_WRT_FIX,'
                            'KLD_SEG_WRT_FIX'
                            '\n')
        self.cc_file = open(join(self.metrics_dir, 'cc.txt'), mode='w')
        self.cc_file.write('FRAME_NUMBER,'
                           'CC_DREYEVE_WRT_SAL,'
                           'CC_IMAGE_WRT_SAL,'
                           'CC_FLOW_WRT_SAL,'
                           'CC_SEG_WRT_SAL,'
                           'CC_DREYEVE_WRT_FIX,'
                           'CC_IMAGE_WRT_FIX,'
                           'CC_FLOW_WRT_FIX,'
                           'CC_SEG_WRT_FIX'
                           '\n')

        # initialize lists to handle values for all frames
        # this is used at the end to compute averages
        self.kld_values = []
        self.cc_values = []

    def feed(self, frame_number, p_dreyeve, p_image, p_flow, p_seg, gt_sal, gt_fix):
        """
        Feeds the saver with new predictions and groundtruth data to evaluate.

        :param frame_number: the index of the frame in evaluation
        :param p_dreyeve: the prediction of the dreyevenet
        :param p_image: the prediction of the image branch
        :param p_flow: the prediction of the optical flow branch
        :param p_seg: the prediction of the segmentation branch
        :param gt_sal: the groundtruth data in terms of saliency map
        :param gt_fix: the groundtruth data in terms of fixation map
        """

        this_frame_kld = [frame_number,
                          kld_numeric(gt_sal, p_dreyeve),
                          kld_numeric(gt_sal, p_image),
                          kld_numeric(gt_sal, p_flow),
                          kld_numeric(gt_sal, p_seg),
                          kld_numeric(gt_fix, p_dreyeve),
                          kld_numeric(gt_fix, p_image),
                          kld_numeric(gt_fix, p_flow),
                          kld_numeric(gt_fix, p_seg)
                          ]

        this_frame_cc = [frame_number,
                         cc_numeric(gt_sal, p_dreyeve),
                         cc_numeric(gt_sal, p_image),
                         cc_numeric(gt_sal, p_flow),
                         cc_numeric(gt_sal, p_seg),
                         cc_numeric(gt_fix, p_dreyeve),
                         cc_numeric(gt_fix, p_image),
                         cc_numeric(gt_fix, p_flow),
                         cc_numeric(gt_fix, p_seg)
                         ]

        self.kld_file.write('{},{},{},{},{},{},{},{},{}\n'.format(*this_frame_kld))
        self.cc_file.write('{},{},{},{},{},{},{},{},{}\n'.format(*this_frame_cc))

        self.kld_values.append(this_frame_kld[1:])  # discard frame number
        self.cc_values.append(this_frame_cc[1:])  # discard frame number

    def save_mean_metrics(self):
        """
        Function to save the mean of the metrics in a separate file.
        """

        with open(join(self.metrics_dir, 'kld_mean.txt'), mode='w') as f:
            f.write('KLD_DREYEVE_WRT_SAL,'
                    'KLD_IMAGE_WRT_SAL,'
                    'KLD_FLOW_WRT_SAL,'
                    'KLD_SEG_WRT_SAL,'
                    'KLD_DREYEVE_WRT_FIX,'
                    'KLD_IMAGE_WRT_FIX,'
                    'KLD_FLOW_WRT_FIX,'
                    'KLD_SEG_WRT_FIX'
                    '\n')
            avg = np.mean(np.array(self.kld_values), axis=0).tolist()
            f.write('{},{},{},{},{},{},{},{}'.format(*avg))

        with open(join(self.metrics_dir, 'cc_mean.txt'), mode='w') as f:
            f.write('CC_DREYEVE_WRT_SAL,'
                    'CC_IMAGE_WRT_SAL,'
                    'CC_FLOW_WRT_SAL,'
                    'CC_SEG_WRT_SAL,'
                    'CC_DREYEVE_WRT_FIX,'
                    'CC_IMAGE_WRT_FIX,'
                    'CC_FLOW_WRT_FIX,'
                    'CC_SEG_WRT_FIX'
                    '\n')
            avg = np.mean(np.array(self.cc_values), axis=0).tolist()
            f.write('{},{},{},{},{},{},{},{}'.format(*avg))


class AblationStudy:
    """
    This class is meant to perform the ablation study. Totally TODO.
    """
    def __init__(self, pred_dir, seq):
        self.ablation_dir = join(pred_dir, '{:02d}'.format(seq), 'ablation')
        if not os.path.exists(self.ablation_dir):
            os.makedirs(self.ablation_dir)

        # open files and put headers in it
        self.kld_file = open(join(self.ablation_dir, 'kld.txt'), mode='w')
        self.kld_file.write('FRAME_NUMBER,'
                            'KLD_DREYEVE_WRT_SAL,'
                            'KLD_IMAGEFLOW_WRT_SAL,'
                            'KLD_IMAGESEG_WRT_SAL,'
                            'KLD_FLOWSEG_WRT_SAL,'
                            'KLD_DREYEVE_WRT_FIX,'
                            'KLD_IMAGEFLOW_WRT_FIX,'
                            'KLD_IMAGESEG_WRT_FIX,'
                            'KLD_FLOWSEG_WRT_FIX'
                            '\n')

        self.cc_file = open(join(self.ablation_dir, 'cc.txt'), mode='w')
        self.cc_file.write('FRAME_NUMBER,'
                           'CC_DREYEVE_WRT_SAL,'
                           'CC_IMAGEFLOW_WRT_SAL,'
                           'CC_IMAGESEG_WRT_SAL,'
                           'CC_FLOWSEG_WRT_SAL,'
                           'CC_DREYEVE_WRT_FIX,'
                           'CC_IMAGEFLOW_WRT_FIX,'
                           'CC_IMAGESEG_WRT_FIX,'
                           'CC_FLOWSEG_WRT_FIX'
                           '\n')

        # initialize lists to handle values for all frames
        # this is used at the end to compute averages
        self.kld_values = []
        self.cc_values = []

    def feed(self, frame_number, p_dreyeve, p_image, p_flow, p_seg, gt_sal, gt_fix):
        """
        Feeds the ablation with new predictions and groundtruth data to evaluate.

        :param frame_number: the index of the frame in evaluation
        :param p_dreyeve: the prediction of the dreyevenet
        :param p_image: the prediction of the image branch
        :param p_flow: the prediction of the optical flow branch
        :param p_seg: the prediction of the segmentation branch
        :param gt_sal: the groundtruth data in terms of saliency map
        :param gt_fix: the groundtruth data in terms of fixation map
        """

        this_frame_kld = [frame_number,
                          kld_numeric(gt_sal, p_dreyeve),
                          kld_numeric(gt_sal, p_image + p_flow),
                          kld_numeric(gt_sal, p_image + p_seg),
                          kld_numeric(gt_sal, p_flow + p_seg),
                          kld_numeric(gt_fix, p_dreyeve),
                          kld_numeric(gt_fix, p_image + p_flow),
                          kld_numeric(gt_fix, p_image + p_seg),
                          kld_numeric(gt_fix, p_flow + p_seg),
                          ]

        this_frame_cc = [frame_number,
                         cc_numeric(gt_sal, p_dreyeve),
                         cc_numeric(gt_sal, p_image + p_flow),
                         cc_numeric(gt_sal, p_image + p_seg),
                         cc_numeric(gt_sal, p_flow + p_seg),
                         cc_numeric(gt_fix, p_dreyeve),
                         cc_numeric(gt_fix, p_image + p_flow),
                         cc_numeric(gt_fix, p_image + p_seg),
                         cc_numeric(gt_fix, p_flow + p_seg),
                         ]

        self.kld_file.write('{},{},{},{},{},{},{},{},{}\n'.format(*this_frame_kld))
        self.cc_file.write('{},{},{},{},{},{},{},{},{}\n'.format(*this_frame_cc))

        self.kld_values.append(this_frame_kld[1:])  # discard frame number
        self.cc_values.append(this_frame_cc[1:])  # discard frame number

    def save_mean_metrics(self):
        """
        Function to save the mean of the metrics in a separate file.
        """

        with open(join(self.ablation_dir, 'kld_mean.txt'), mode='w') as f:
            f.write('KLD_DREYEVE_WRT_SAL,'
                    'KLD_IMAGEFLOW_WRT_SAL,'
                    'KLD_IMAGESEG_WRT_SAL,'
                    'KLD_FLOWSEG_WRT_SAL,'
                    'KLD_DREYEVE_WRT_FIX,'
                    'KLD_IMAGEFLOW_WRT_FIX,'
                    'KLD_IMAGESEG_WRT_FIX,'
                    'KLD_FLOWSEG_WRT_FIX'
                    '\n')
            avg = np.mean(np.array(self.kld_values), axis=0).tolist()
            f.write('{},{},{},{},{},{},{},{}'.format(*avg))

        with open(join(self.ablation_dir, 'cc_mean.txt'), mode='w') as f:
            f.write('CC_DREYEVE_WRT_SAL,'
                    'CC_IMAGEFLOW_WRT_SAL,'
                    'CC_IMAGESEG_WRT_SAL,'
                    'CC_FLOWSEG_WRT_SAL,'
                    'CC_DREYEVE_WRT_FIX,'
                    'CC_IMAGEFLOW_WRT_FIX,'
                    'CC_IMAGESEG_WRT_FIX,'
                    'CC_FLOWSEG_WRT_FIX'
                    '\n')
            avg = np.mean(np.array(self.cc_values), axis=0).tolist()
            f.write('{},{},{},{},{},{},{},{}'.format(*avg))


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--stop")
    args = parser.parse_args()

    start_seq = 1 if args.start is None else int(args.start)
    stop_seq = 74 if args.stop is None else int(args.stop)
    sequences = range(start_seq, stop_seq + 1)

    # some variables
    frames_per_seq, h, w = 16, 448, 448

    pred_dir = 'Z:\PREDICTIONS_2017'
    dreyeve_dir = 'Z:\DATA'

    for seq in sequences:
        print 'Processing sequence {}'.format(seq)

        # prediction dirs
        dir_pred_dreyevenet = join(pred_dir, '{:02d}'.format(seq), 'dreyeveNet')
        dir_pred_image = join(pred_dir, '{:02d}'.format(seq), 'image_branch')
        dir_pred_flow = join(pred_dir, '{:02d}'.format(seq), 'flow_branch')
        dir_pred_seg = join(pred_dir, '{:02d}'.format(seq), 'semseg_branch')

        # gt dirs
        dir_gt_sal = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        dir_gt_fix = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print 'Computing metrics...'
        metric_saver = MetricSaver(pred_dir, seq)
        ablation = AblationStudy(pred_dir, seq)

        for fr in tqdm(xrange(15, 7500-1)):
            # load predictions
            p_dreyeve = np.squeeze(np.load(join(dir_pred_dreyevenet, '{:06}.npz'.format(fr)))['arr_0'])
            p_image = np.squeeze(np.load(join(dir_pred_image, '{:06}.npz'.format(fr)))['arr_0'])
            p_flow = np.squeeze(np.load(join(dir_pred_flow, '{:06}.npz'.format(fr)))['arr_0'])
            p_seg = np.squeeze(np.load(join(dir_pred_seg, '{:06}.npz'.format(fr)))['arr_0'])

            # load gts
            # todo is this correct? should I resize groundtruth or predictions?
            gt_sal = read_image(join(dir_gt_sal, '{:06d}.png'.format(fr)), channels_first=False,
                                color=False, resize_dim=(h, w))
            gt_fix = read_image(join(dir_gt_fix, '{:06d}.png'.format(fr)), channels_first=False,
                                color=False, resize_dim=(h, w))

            # feed the saver
            metric_saver.feed(fr, p_dreyeve, p_image, p_flow, p_seg, gt_sal, gt_fix)
            ablation.feed(fr, p_dreyeve, p_image, p_flow, p_seg, gt_sal, gt_fix)

        # save mean values
        metric_saver.save_mean_metrics()
        ablation.save_mean_metrics()
