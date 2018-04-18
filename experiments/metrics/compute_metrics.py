from __future__ import print_function
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from glob import glob
from computer_vision_utils.io_helper import read_image
from metrics.metrics import kld_numeric
from metrics.metrics import cc_numeric
from metrics.metrics import ig_numeric


class MetricSaver:
    """
    This class handles metric computation given predictions and groundtruth.

    Params:
    pred_dir: the prediction directory
    seq: the number of the sequence
    model: choose among ['old', 'new', 'central_gaussian', 'mean_gt', 'competitor']
    """
    def __init__(self, pred_dir, seq, model):
        models = ['old', 'new', 'central_gaussian', 'mean_gt', 'competitor']
        assert model in models, 'model should be one of {}'.format(models)

        self.model = model

        # create metrics dir
        self.metrics_dir = join(pred_dir, '{:02d}'.format(seq), 'metrics')
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

        # build headers
        if self.model == 'old':
            self.kld_header = ['FRAME_NUMBER,',
                               'KLD_WRT_SAL,',
                               'KLD_WRT_FIX,',
                               '\n']

            self.cc_header = ['FRAME_NUMBER,',
                              'CC_WRT_SAL,',
                              'CC_WRT_FIX,',
                              '\n']

            self.ig_header = ['FRAME_NUMBER,',
                              'IG'
                              '\n']
        elif self.model == 'new':
            self.kld_header = ['FRAME_NUMBER,',
                               'KLD_DREYEVE_WRT_SAL,',
                               'KLD_IMAGE_WRT_SAL,',
                               'KLD_FLOW_WRT_SAL,',
                               'KLD_SEG_WRT_SAL,',
                               'KLD_DREYEVE_WRT_FIX,',
                               'KLD_IMAGE_WRT_FIX,',
                               'KLD_FLOW_WRT_FIX,',
                               'KLD_SEG_WRT_FIX',
                               '\n']

            self.cc_header = ['FRAME_NUMBER,',
                              'CC_DREYEVE_WRT_SAL,',
                              'CC_IMAGE_WRT_SAL,',
                              'CC_FLOW_WRT_SAL,',
                              'CC_SEG_WRT_SAL,',
                              'CC_DREYEVE_WRT_FIX,',
                              'CC_IMAGE_WRT_FIX,',
                              'CC_FLOW_WRT_FIX,',
                              'CC_SEG_WRT_FIX',
                              '\n']

            self.ig_header = ['FRAME_NUMBER,',
                              'IG_DREYEVE,',
                              'IG_IMAGE,',
                              'IG_FLOW,',
                              'IG_SEG',
                              '\n']

        elif self.model == 'central_gaussian':
            self.kld_header = ['FRAME_NUMBER,',
                               'KLD_WRT_SAL,',
                               'KLD_WRT_FIX,',
                               '\n']

            self.cc_header = ['FRAME_NUMBER,',
                              'CC_WRT_SAL,',
                              'CC_WRT_FIX,',
                              '\n']

            self.ig_header = ['FRAME_NUMBER,',
                              'IG'
                              '\n']

        elif self.model == 'mean_gt':
            self.kld_header = ['FRAME_NUMBER,',
                               'KLD_WRT_SAL,',
                               'KLD_WRT_FIX,',
                               '\n']

            self.cc_header = ['FRAME_NUMBER,',
                              'CC_WRT_SAL,',
                              'CC_WRT_FIX,',
                              '\n']

            self.ig_header = ['FRAME_NUMBER,',
                              'IG'
                              '\n']

        elif self.model == 'competitor':
            self.kld_header = ['FRAME_NUMBER,',
                               'KLD_WRT_SAL,',
                               'KLD_WRT_FIX,',
                               '\n']

            self.cc_header = ['FRAME_NUMBER,',
                              'CC_WRT_SAL,',
                              'CC_WRT_FIX,',
                              '\n']

            self.ig_header = ['FRAME_NUMBER,',
                              'IG'
                              '\n']

        # open files and put headers in it
        self.kld_file = open(join(self.metrics_dir, 'kld.txt'), mode='w')
        self.kld_file.write(('{}'*len(self.kld_header)).format(*self.kld_header))

        self.cc_file = open(join(self.metrics_dir, 'cc.txt'), mode='w')
        self.cc_file.write(('{}'*len(self.cc_header)).format(*self.cc_header))

        self.ig_file = open(join(self.metrics_dir, 'ig.txt'), mode='w')
        self.ig_file.write(('{}'*len(self.ig_header)).format(*self.ig_header))

        # initialize lists to handle values for all frames
        # this is used at the end to compute averages
        self.kld_values = []
        self.cc_values = []
        self.ig_values = []

    def feed(self, frame_number, predictions, groundtruth, ig_baseline):
        """
        Feeds the saver with new predictions and groundtruth data to evaluate.

        :param frame_number: the index of the frame in evaluation.
        :param predictions: a list of numpy array encoding predictions.
            If self.model in ['old', 'central_gaussian', 'competitor'], a singleton list like [model_prediction] is expected.
            If self.model == 'new' a list like [p_dreyeve, p_image, p_flow, p_seg] is expected.
            If self.model == 'mean_gt' a list like [mean_gt_sal, mean_gt_fix] is expected.
        :param groundtruth: a list like [gt_sal, gt_fix]
        :param ig_baseline: a prediction used as baseline for information gain metric.
        """

        gt_sal, gt_fix = groundtruth

        if self.model == 'old':
            p = predictions[0]

            this_frame_kld = [frame_number,
                              kld_numeric(gt_sal, p),
                              kld_numeric(gt_fix, p),
                              ]

            this_frame_cc = [frame_number,
                             cc_numeric(gt_sal, p),
                             cc_numeric(gt_fix, p),
                             ]

            this_frame_ig = [frame_number,
                             ig_numeric(gt_fix, p, ig_baseline)]

        elif self.model == 'new':
            p_dreyeve, p_image, p_flow, p_seg = predictions

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

            this_frame_ig = [frame_number,
                             ig_numeric(gt_fix, p_dreyeve, ig_baseline),
                             ig_numeric(gt_fix, p_image, ig_baseline),
                             ig_numeric(gt_fix, p_flow, ig_baseline),
                             ig_numeric(gt_fix, p_seg, ig_baseline)]

        if self.model == 'central_gaussian':
            p = predictions[0]

            this_frame_kld = [frame_number,
                              kld_numeric(gt_sal, p),
                              kld_numeric(gt_fix, p),
                              ]

            this_frame_cc = [frame_number,
                             cc_numeric(gt_sal, p),
                             cc_numeric(gt_fix, p),
                             ]

            this_frame_ig = [frame_number,
                             ig_numeric(gt_fix, p, ig_baseline)]

        if self.model == 'mean_gt':
            mean_gt_sal, mean_gt_fix = predictions

            this_frame_kld = [frame_number,
                              kld_numeric(gt_sal, mean_gt_sal),
                              kld_numeric(gt_fix, mean_gt_fix),
                              ]

            this_frame_cc = [frame_number,
                             cc_numeric(gt_sal, mean_gt_sal),
                             cc_numeric(gt_fix, mean_gt_fix),
                             ]

            this_frame_ig = [frame_number,
                             ig_numeric(gt_fix, mean_gt_fix, ig_baseline)]  # doesn't make sense in this case

        if self.model == 'competitor':
            p = predictions[0]

            this_frame_kld = [frame_number,
                              kld_numeric(gt_sal, p),
                              kld_numeric(gt_fix, p),
                              ]

            this_frame_cc = [frame_number,
                             cc_numeric(gt_sal, p),
                             cc_numeric(gt_fix, p),
                             ]

            this_frame_ig = [frame_number,
                             ig_numeric(gt_fix, p, ig_baseline)]

        self.kld_file.write(('{},' * len(this_frame_kld) + '\n').format(*this_frame_kld))
        self.cc_file.write(('{},' * len(this_frame_cc) + '\n').format(*this_frame_cc))
        self.ig_file.write(('{},' * len(this_frame_ig) + '\n').format(*this_frame_ig))

        self.kld_values.append(this_frame_kld[1:])  # discard frame number
        self.cc_values.append(this_frame_cc[1:])  # discard frame number
        self.ig_values.append(this_frame_ig[1:])  # discard frame number

    def save_mean_metrics(self):
        """
        Function to save the mean of the metrics in a separate file.
        """

        with open(join(self.metrics_dir, 'kld_mean.txt'), mode='w') as f:
            header = self.kld_header[1:]  # discard frame number
            f.write(('{}'*len(header)).format(*header))

            # compute average and save
            avg = np.nanmean(np.array(self.kld_values), axis=0).tolist()
            f.write(('{},'*len(avg)).format(*avg))

        with open(join(self.metrics_dir, 'cc_mean.txt'), mode='w') as f:
            header = self.cc_header[1:]  # discard frame number
            f.write(('{}'*len(header)).format(*header))

            # compute average and save
            avg = np.nanmean(np.array(self.cc_values), axis=0).tolist()
            f.write(('{},'*len(avg)).format(*avg))

        with open(join(self.metrics_dir, 'ig_mean.txt'), mode='w') as f:
            header = self.ig_header[1:]  # discard frame number
            f.write(('{}'*len(header)).format(*header))

            # compute average and save
            avg = np.nanmean(np.array(self.ig_values), axis=0).tolist()
            f.write(('{},'*len(avg)).format(*avg))


class AblationStudy:
    """
    This class is meant to perform the ablation study. Ablation study makes sense
    only in the new model with three branches.

    Params:
        pred_dir: the prediction directory
        seq: the number of the sequence
    """
    def __init__(self, pred_dir, seq):

        # create ablation dir
        self.ablation_dir = join(pred_dir, '{:02d}'.format(seq), 'ablation')
        if not os.path.exists(self.ablation_dir):
            os.makedirs(self.ablation_dir)

        # build headers
        self.kld_header = ['FRAME_NUMBER,',
                           'KLD_DREYEVE_WRT_SAL,',
                           'KLD_IMAGEFLOW_WRT_SAL,',
                           'KLD_IMAGESEG_WRT_SAL,',
                           'KLD_FLOWSEG_WRT_SAL,',
                           'KLD_DREYEVE_WRT_FIX,',
                           'KLD_IMAGEFLOW_WRT_FIX,',
                           'KLD_IMAGESEG_WRT_FIX,',
                           'KLD_FLOWSEG_WRT_FIX',
                           '\n']

        self.cc_header = ['FRAME_NUMBER,',
                          'CC_DREYEVE_WRT_SAL,',
                          'CC_IMAGEFLOW_WRT_SAL,',
                          'CC_IMAGESEG_WRT_SAL,',
                          'CC_FLOWSEG_WRT_SAL,',
                          'CC_DREYEVE_WRT_FIX,',
                          'CC_IMAGEFLOW_WRT_FIX,',
                          'CC_IMAGESEG_WRT_FIX,',
                          'CC_FLOWSEG_WRT_FIX',
                          '\n']

        self.ig_header = ['FRAME_NUMBER,',
                          'IG_DREYEVE,',
                          'IG_IMAGEFLOW,',
                          'IG_IMAGESEG,',
                          'IG_FLOWSEG',
                          '\n']

        # open files and put headers in it
        self.kld_file = open(join(self.ablation_dir, 'kld.txt'), mode='w')
        self.kld_file.write(('{}' * len(self.kld_header)).format(*self.kld_header))

        self.cc_file = open(join(self.ablation_dir, 'cc.txt'), mode='w')
        self.cc_file.write(('{}' * len(self.cc_header)).format(*self.cc_header))

        self.ig_file = open(join(self.ablation_dir, 'ig.txt'), mode='w')
        self.ig_file.write(('{}'*len(self.ig_header)).format(*self.ig_header))

        # initialize lists to handle values for all frames
        # this is used at the end to compute averages
        self.kld_values = []
        self.cc_values = []
        self.ig_values = []

    def feed(self, frame_number, predictions, groundtruth, ig_baseline):
        """
        Feeds the ablation with new predictions and groundtruth data to evaluate.

        :param frame_number: the index of the frame in evaluation
        :param predictions: a list of numpy array encoding predictions.
            A list like [p_dreyeve, p_image, p_flow, p_seg] is expected.
        :param groundtruth: a list like [gt_sal, gt_fix]
        :param ig_baseline: a prediction used as baseline for information gain metric.
        """

        p_dreyeve, p_image, p_flow, p_seg = predictions
        gt_sal, gt_fix = groundtruth

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

        this_frame_ig = [frame_number,
                         ig_numeric(gt_fix, p_dreyeve, ig_baseline),
                         ig_numeric(gt_fix, p_image + p_flow, ig_baseline),
                         ig_numeric(gt_fix, p_image + p_seg, ig_baseline),
                         ig_numeric(gt_fix, p_flow + p_seg, ig_baseline)]

        self.kld_file.write(('{},' * len(this_frame_kld) + '\n').format(*this_frame_kld))
        self.cc_file.write(('{},' * len(this_frame_cc) + '\n').format(*this_frame_cc))
        self.ig_file.write(('{},' * len(this_frame_ig) + '\n').format(*this_frame_ig))

        self.kld_values.append(this_frame_kld[1:])  # discard frame number
        self.cc_values.append(this_frame_cc[1:])  # discard frame number
        self.ig_values.append(this_frame_ig[1:])  # discard frame number

    def save_mean_metrics(self):
        """
        Function to save the mean of the metrics in a separate file.
        """

        with open(join(self.ablation_dir, 'kld_mean.txt'), mode='w') as f:
            header = self.kld_header[1:]  # discard frame number
            f.write(('{}' * len(header)).format(*header))

            # compute average and save
            avg = np.nanmean(np.array(self.kld_values), axis=0).tolist()
            f.write(('{},'*len(avg)).format(*avg))

        with open(join(self.ablation_dir, 'cc_mean.txt'), mode='w') as f:
            header = self.cc_header[1:]  # discard frame number
            f.write(('{}' * len(header)).format(*header))

            # compute average and save
            avg = np.nanmean(np.array(self.cc_values), axis=0).tolist()
            f.write(('{},'*len(avg)).format(*avg))

        with open(join(self.ablation_dir, 'ig_mean.txt'), mode='w') as f:
            header = self.ig_header[1:]  # discard frame number
            f.write(('{}'*len(header)).format(*header))

            # compute average and save
            avg = np.nanmean(np.array(self.ig_values), axis=0).tolist()
            f.write(('{},'*len(avg)).format(*avg))


def compute_metrics_for_central_crop(sequences):
    """
    Function to compute metrics out of the new model trained with central crop only (2018).

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = '/majinbu/public/DREYEVE/PREDICTIONS_CENTRAL_CROP'
    dreyeve_dir = '/majinbu/public/DREYEVE/DATA'

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    for seq in sequences:
        print('Processing sequence {}'.format(seq))

        # prediction dirs
        dir_pred_dreyevenet = join(pred_dir, '{:02d}'.format(seq), 'dreyeveNet')
        dir_pred_image = join(pred_dir, '{:02d}'.format(seq), 'image_branch')
        dir_pred_flow = join(pred_dir, '{:02d}'.format(seq), 'flow_branch')
        dir_pred_seg = join(pred_dir, '{:02d}'.format(seq), 'semseg_branch')

        # gt dirs
        dir_gt_sal = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        dir_gt_fix = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print('Computing metrics...')
        metric_saver = MetricSaver(pred_dir, seq, model='new')
        ablation = AblationStudy(pred_dir, seq)

        for fr in tqdm(xrange(15, 7500 - 1, 5)):
            # load predictions
            p_dreyeve = np.squeeze(np.load(join(dir_pred_dreyevenet, '{:06}.npz'.format(fr)))['arr_0'])
            p_image = np.squeeze(np.load(join(dir_pred_image, '{:06}.npz'.format(fr)))['arr_0'])
            p_flow = np.squeeze(np.load(join(dir_pred_flow, '{:06}.npz'.format(fr)))['arr_0'])
            p_seg = np.squeeze(np.load(join(dir_pred_seg, '{:06}.npz'.format(fr)))['arr_0'])

            p_dreyeve = cv2.resize(p_dreyeve, dsize=(gt_h, gt_w)[::-1])
            p_image = cv2.resize(p_image, dsize=(gt_h, gt_w)[::-1])
            p_flow = cv2.resize(p_flow, dsize=(gt_h, gt_w)[::-1])
            p_seg = cv2.resize(p_seg, dsize=(gt_h, gt_w)[::-1])

            # load gts
            gt_sal = read_image(join(dir_gt_sal, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)
            gt_fix = read_image(join(dir_gt_fix, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)

            # feed the saver
            metric_saver.feed(fr, predictions=[p_dreyeve, p_image, p_flow, p_seg], groundtruth=[gt_sal, gt_fix],
                              ig_baseline=ig_baseline)
            ablation.feed(fr, predictions=[p_dreyeve, p_image, p_flow, p_seg], groundtruth=[gt_sal, gt_fix],
                          ig_baseline=ig_baseline)

        # save mean values
        metric_saver.save_mean_metrics()
        ablation.save_mean_metrics()


def compute_metrics_for_new_model(sequences):
    """
    Function to compute metrics out of the new model (2017).

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_2017'
    dreyeve_dir = 'Z:\\DATA'

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    for seq in sequences:
        print('Processing sequence {}'.format(seq))

        # prediction dirs
        dir_pred_dreyevenet = join(pred_dir, '{:02d}'.format(seq), 'dreyeveNet')
        dir_pred_image = join(pred_dir, '{:02d}'.format(seq), 'image_branch')
        dir_pred_flow = join(pred_dir, '{:02d}'.format(seq), 'flow_branch')
        dir_pred_seg = join(pred_dir, '{:02d}'.format(seq), 'semseg_branch')

        # gt dirs
        dir_gt_sal = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        dir_gt_fix = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print('Computing metrics...')
        metric_saver = MetricSaver(pred_dir, seq, model='new')
        ablation = AblationStudy(pred_dir, seq)

        for fr in tqdm(xrange(15, 7500 - 1, 5)):
            # load predictions
            p_dreyeve = np.squeeze(np.load(join(dir_pred_dreyevenet, '{:06}.npz'.format(fr)))['arr_0'])
            p_image = np.squeeze(np.load(join(dir_pred_image, '{:06}.npz'.format(fr)))['arr_0'])
            p_flow = np.squeeze(np.load(join(dir_pred_flow, '{:06}.npz'.format(fr)))['arr_0'])
            p_seg = np.squeeze(np.load(join(dir_pred_seg, '{:06}.npz'.format(fr)))['arr_0'])

            p_dreyeve = cv2.resize(p_dreyeve, dsize=(gt_h, gt_w)[::-1])
            p_image = cv2.resize(p_image, dsize=(gt_h, gt_w)[::-1])
            p_flow = cv2.resize(p_flow, dsize=(gt_h, gt_w)[::-1])
            p_seg = cv2.resize(p_seg, dsize=(gt_h, gt_w)[::-1])

            # load gts
            gt_sal = read_image(join(dir_gt_sal, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)
            gt_fix = read_image(join(dir_gt_fix, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)

            # feed the saver
            metric_saver.feed(fr, predictions=[p_dreyeve, p_image, p_flow, p_seg], groundtruth=[gt_sal, gt_fix],
                              ig_baseline=ig_baseline)
            ablation.feed(fr, predictions=[p_dreyeve, p_image, p_flow, p_seg], groundtruth=[gt_sal, gt_fix],
                          ig_baseline=ig_baseline)

        # save mean values
        metric_saver.save_mean_metrics()
        ablation.save_mean_metrics()


def compute_metrics_for_old_model(sequences):
    """
    Function to compute metrics out of the old model (2015).

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS\\architecture7'
    dreyeve_dir = 'Z:\\DATA'

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                             resize_dim=(gt_h, gt_w))

    for seq in sequences:
        print('Processing sequence {}'.format(seq))

        # prediction dirs
        seq_pred_dir = join(pred_dir, '{:02d}'.format(seq), 'output')

        # gt dirs
        seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print('Computing metrics...')
        metric_saver = MetricSaver(pred_dir, seq, model='old')

        for fr in tqdm(xrange(15, 7500 - 1, 5)):
            # load predictions
            p = read_image(join(seq_pred_dir, '{:06}.png'.format(fr+1)), channels_first=False, color=False,
                           resize_dim=(gt_h, gt_w))

            # load gts
            gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)
            gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)

            # feed the saver
            metric_saver.feed(fr, predictions=[p], groundtruth=[gt_sal, gt_fix], ig_baseline=ig_baseline)

        # save mean values
        metric_saver.save_mean_metrics()


def compute_metrics_for_mlnet_model(sequences):
    """
    Function to compute metrics out of the mlnet model (Marcy).

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_MLNET'
    dreyeve_dir = 'Z:\\DATA'

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    for seq in sequences:
        print('Processing sequence {}'.format(seq))

        # prediction dirs
        seq_pred_dir = join(pred_dir, '{:02d}'.format(seq), 'output')

        # gt dirs
        seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print('Computing metrics...')
        metric_saver = MetricSaver(pred_dir, seq, model='old')

        for fr in tqdm(xrange(15, 7500 - 1, 5)):
            # load predictions
            p = read_image(join(seq_pred_dir, '{:06}.png'.format(fr+1)), channels_first=False, color=False,
                           resize_dim=(gt_h, gt_w))

            # load gts
            gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)
            gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)

            # feed the saver
            metric_saver.feed(fr, predictions=[p], groundtruth=[gt_sal, gt_fix], ig_baseline=ig_baseline)

        # save mean values
        metric_saver.save_mean_metrics()


def compute_metrics_for_rmdn_model(sequences):
    """
    Function to compute metrics out of the RMDN model [4].

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_RMDN'
    dreyeve_dir = 'Z:\\DATA'

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    for seq in sequences:
        print('Processing sequence {}'.format(seq))

        # prediction dirs
        seq_pred_dir = join(pred_dir, '{:02d}'.format(seq), 'output')

        # gt dirs
        seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print('Computing metrics...')
        metric_saver = MetricSaver(pred_dir, seq, model='old')

        for fr in tqdm(xrange(15, 7500 - 1, 5)):
            # load predictions
            p = read_image(join(seq_pred_dir, '{:06}.png'.format(fr+1)), channels_first=False, color=False,
                           resize_dim=(gt_h, gt_w))

            # load gts
            gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)
            gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)

            # feed the saver
            metric_saver.feed(fr, predictions=[p], groundtruth=[gt_sal, gt_fix], ig_baseline=ig_baseline)

        # save mean values
        metric_saver.save_mean_metrics()


def compute_metrics_for_central_gaussian(sequences):
    """
    Function to compute metrics using a central gaussian as prediction.

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_CENTRAL_GAUSSIAN'
    dreyeve_dir = 'Z:\\DATA'

    central_gaussian = read_image(join(dreyeve_dir, 'gaussian_baseline.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    for seq in sequences:
        print('Processing sequence {}'.format(seq))

        # gt dirs
        seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print('Computing metrics...')
        metric_saver = MetricSaver(pred_dir, seq, model='central_gaussian')

        for fr in tqdm(xrange(15, 7500 - 1, 5)):
            # load gts
            gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)
            gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)

            # feed the saver
            metric_saver.feed(fr, predictions=[central_gaussian], groundtruth=[gt_sal, gt_fix], ig_baseline=ig_baseline)

        # save mean values
        metric_saver.save_mean_metrics()


def compute_metrics_for_mean_gt(sequences):
    """
    Function to compute metrics using the mean training gt as prediction.

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_MEAN_TRAIN_GT'
    dreyeve_dir = 'Z:\\DATA'

    mean_gt_sal = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))
    mean_gt_fix = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    for seq in sequences:
        print('Processing sequence {}'.format(seq))

        # gt dirs
        seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
        seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        print('Computing metrics...')
        metric_saver = MetricSaver(pred_dir, seq, model='mean_gt')

        for fr in tqdm(xrange(15, 7500 - 1, 5)):
            # load gts
            gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)
            gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                                color=False)

            # feed the saver
            metric_saver.feed(fr, predictions=[mean_gt_sal, mean_gt_fix], groundtruth=[gt_sal, gt_fix],
                              ig_baseline=ig_baseline)

        # save mean values
        metric_saver.save_mean_metrics()


def compute_metrics_for_wang2015consistent(sequences):
    """
    Function to compute metrics from model [1].

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_wang2015consistent'
    dreyeve_dir = 'Z:\\DATA'

    pred_list = glob('Z:/CODE_EXPERIMENTS/EXP_VIDEOSALIENCY_COMPARISON/methods/videosal-master/final_results/**/saliency/*.bmp')

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    print('Computing metrics...')
    last_seq = 0
    for index in tqdm(xrange(0, len(pred_list))):
        pred_img = pred_list[index]

        # recover sequence number
        seq = int(os.path.basename(os.path.dirname(os.path.dirname(pred_img))).split('_')[1])

        # if sequence is new, recover some variables
        if seq != last_seq:    # save mean values
            if 'metric_saver' in locals():
                metric_saver.save_mean_metrics()

            metric_saver = MetricSaver(pred_dir, seq, model='competitor')
            last_seq = seq

            # gt dirs
            seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
            seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        # get frame number
        fr = int(os.path.basename(pred_img).split('.')[0])

        # load prediction
        p = read_image(pred_img, channels_first=False, color=False, resize_dim=(gt_h, gt_w))

        # load gts
        gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                            color=False)
        gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                            color=False)

        # feed the saver
        metric_saver.feed(fr, predictions=[p], groundtruth=[gt_sal, gt_fix], ig_baseline=ig_baseline)
    metric_saver.save_mean_metrics()


def compute_metrics_for_wang2015saliency(sequences):
    """
    Function to compute metrics from model [2].

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_wang2015saliency'
    dreyeve_dir = 'Z:\\DATA'

    pred_list = glob('Z:/CODE_EXPERIMENTS/EXP_VIDEOSALIENCY_COMPARISON/methods/SaliencySeg-master/code/final_results/**/final_saliency/*.bmp')

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    print('Computing metrics...')
    last_seq = 0
    for index in tqdm(xrange(0, len(pred_list))):
        pred_img = pred_list[index]

        # recover sequence number
        seq = int(os.path.basename(os.path.dirname(os.path.dirname(pred_img))).split('_')[1])

        # if sequence is new, recover some variables
        if seq != last_seq:
            if 'metric_saver' in locals():
                metric_saver.save_mean_metrics()

            metric_saver = MetricSaver(pred_dir, seq, model='competitor')
            last_seq = seq

            # gt dirs
            seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
            seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        # get frame number
        fr = int(os.path.basename(pred_img).split('.')[0])

        # load prediction
        p = read_image(pred_img, channels_first=False, color=False, resize_dim=(gt_h, gt_w))

        # load gts
        gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                            color=False)
        gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                            color=False)

        # feed the saver
        metric_saver.feed(fr, predictions=[p], groundtruth=[gt_sal, gt_fix], ig_baseline=ig_baseline)

    # save mean values
    metric_saver.save_mean_metrics()


def compute_metrics_for_mathe(sequences):
    """
    Function to compute metrics from model [3].

    :param sequences: A list of sequences to consider.
    """

    # some variables
    gt_h, gt_w = 1080, 1920

    pred_dir = 'Z:\\PREDICTIONS_mathe'
    dreyeve_dir = 'Z:\\DATA'

    pred_list = glob('Z:/CODE_EXPERIMENTS/EXP_VIDEOSALIENCY_COMPARISON/methods/action_in_the_eye/RESULT/**/*.png')

    ig_baseline = read_image(join(dreyeve_dir, 'dreyeve_mean_train_gt_fix.png'), channels_first=False, color=False,
                                  resize_dim=(gt_h, gt_w))

    print('Computing metrics...')
    last_seq = 0
    for index in tqdm(xrange(0, len(pred_list))):
        pred_img = pred_list[index]

        # recover sequence number
        seq = int(os.path.basename(os.path.dirname(pred_img)).split('_')[0])

        # if sequence is new, recover some variables
        if seq != last_seq:
            if 'metric_saver' in locals():
                metric_saver.save_mean_metrics()

            metric_saver = MetricSaver(pred_dir, seq, model='competitor')
            last_seq = seq

            # gt dirs
            seq_gt_sal_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency')
            seq_gt_fix_dir = join(dreyeve_dir, '{:02d}'.format(seq), 'saliency_fix')

        # get frame number
        fr = int(os.path.basename(pred_img).split('.')[0])

        # load prediction
        p = read_image(pred_img, channels_first=False, color=False, resize_dim=(gt_h, gt_w))

        # load gts
        gt_sal = read_image(join(seq_gt_sal_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                            color=False)
        gt_fix = read_image(join(seq_gt_fix_dir, '{:06d}.png'.format(fr+1)), channels_first=False,
                            color=False)

        # feed the saver
        metric_saver.feed(fr, predictions=[p], groundtruth=[gt_sal, gt_fix], ig_baseline=ig_baseline)

    # save mean values
    metric_saver.save_mean_metrics()


if __name__ == '__main__':
    # exit()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--stop")
    args = parser.parse_args()

    start_seq = 1 if args.start is None else int(args.start)
    stop_seq = 74 if args.stop is None else int(args.stop)
    sequences = range(start_seq, stop_seq + 1)

    compute_metrics_for_central_crop(sequences)


"""
REFERENCES:
[1] Wang, Wenguan, Jianbing Shen, and Ling Shao. "Consistent video saliency using local gradient
flow optimization and global refinement." IEEE Transactions on Image Processing 24.11 (2015): 4185-4196.

[2] Wang, Wenguan, Jianbing Shen, and Fatih Porikli. "Saliency-aware geodesic video object segmentation."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

[3] Mathe, Stefan, and Cristian Sminchisescu. "Actions in the eye: dynamic gaze datasets and learnt
saliency models for visual recognition." IEEE transactions on pattern analysis and machine intelligence
37.7 (2015): 1408-1424.

[4] Bazzani, Loris and Larochelle, Hugo and Torresani, Lorenzo. "Recurrent Mixture Density
Network for Spatiotemporal Visual Attention" International Conference on Learning Representations (ICLR 2017)

"""