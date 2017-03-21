import cv2
import numpy as np
from os.path import exists, join
import csv
import glob
import os
import csv


if __name__ == '__main__':

    # read file containing manually annotated subsequences
    subsequences = []
    subseq_file = '//quesada/f/DREYEVE/subsequences.txt'
    with open(subseq_file, 'rb') as csv_subseq:
        csvreader = csv.reader(csv_subseq, delimiter='\t')
        subsequences = list(csvreader)

    cc_f_by_f,  kl_f_by_f   = [], []
    cc_all,     kl_all      = [], []

    prediction_root = '//quesada/f/DREYEVE/PREDICTIONS_2017'
    test_runs_measures = [(r, join(prediction_root, '{:02d}'.format(r))) for r in range(38, 75)]

    for (r, run_dir) in test_runs_measures:

        with open(join(run_dir, 'metrics', 'cc.txt'), 'rb') as f:
            reader = csv.reader(f)
            headers = reader.next()  # store headers
            cc_f_by_f = np.array([map(np.float32, row) for row in reader])

        with open(join(run_dir, 'metrics', 'kld.txt'), 'r') as f:
            reader = csv.reader(f)
            headers = reader.next()  # store headers
            kl_f_by_f = np.array([map(np.float32, row) for row in reader])

        # make a list with all the frames of this run marked as 'keep'
        # each row of `keep_this_run` is a list of four entries [num_run, start_frame, end_frame, subsequence_class]
        keep_this_run = [l for l in subsequences if l[0] == '{0}'.format(r) and l[3] == 'k']
        ranges = [range(int(r[1]), int(r[2])+1) for r in keep_this_run]
        frames_keep_this_run = [rr for r in ranges for rr in r]
        frames_keep_this_run = [f-1 for f in frames_keep_this_run]  # from matlab to python indexing (0-based)
        frames_keep_this_run = set(frames_keep_this_run)

        # keep only metrics for frames in the selected list (these are frames marked as attentive)
        filtered_cc = np.array([cc_row for cc_row in cc_f_by_f if int(cc_row[0]) in frames_keep_this_run])
        filtered_kl = np.array([kl_row for kl_row in kl_f_by_f if int(kl_row[0]) in frames_keep_this_run])

        mean_cc_attentive_this_run = np.mean(filtered_cc, axis=0)[1:]  # skip first col that is the frame index
        mean_kl_attentive_this_run = np.mean(filtered_kl, axis=0)[1:]  # skip first col that is the frame index

        cc_all.append(mean_cc_attentive_this_run)
        kl_all.append(mean_kl_attentive_this_run)

    # at the end, average results over all test runs
    mean_all_cc_attentive = np.mean(np.array(cc_all), axis=0)
    mean_all_kl_attentive = np.mean(np.array(kl_all), axis=0)

    print('CC_DREYEVE_WRT_SAL, CC_IMAGE_WRT_SAL, CC_FLOW_WRT_SAL, CC_SEG_WRT_SAL')
    print(mean_all_cc_attentive[:4])
    print('CC_DREYEVE_WRT_FIX, CC_IMAGE_WRT_FIX, CC_FLOW_WRT_FIX, CC_SEG_WRT_FIX')
    print(mean_all_cc_attentive[4:])
    print('')
    print('KLD_DREYEVE_WRT_SAL,KLD_IMAGE_WRT_SAL,KLD_FLOW_WRT_SAL,KLD_SEG_WRT_SAL')
    print(mean_all_kl_attentive[:4])
    print('KLD_DREYEVE_WRT_FIX,KLD_IMAGE_WRT_FIX,KLD_FLOW_WRT_FIX,KLD_SEG_WRT_FIX')
    print(mean_all_kl_attentive[4:])

