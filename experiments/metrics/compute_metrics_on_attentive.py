from __future__ import print_function
import csv
import numpy as np
from os.path import join


if __name__ == '__main__':

    # read file containing manually annotated subsequences
    subsequences = []
    subseq_file = '/majinbu/public/DREYEVE/subsequences.txt'
    with open(subseq_file, 'rb') as csv_subseq:
        csvreader = csv.reader(csv_subseq, delimiter='\t')
        subsequences = list(csvreader)

    cc_all, kl_all, ig_all = [], [], []

    prediction_root = '/majinbu/public/DREYEVE/PREDICTIONS_DISPLACED'
    test_runs_measures = [(r, join(prediction_root, '{:02d}'.format(r))) for r in range(38, 75)]

    for (r, run_dir) in test_runs_measures:

        with open(join(run_dir, 'metrics', 'cc.txt'), 'rb') as f:
            reader = csv.reader(f)
            headers = reader.next()  # store headers
            cc_f_by_f = {int(row[0]): np.array(row[1:], dtype=np.float32) for row in reader}

        with open(join(run_dir, 'metrics', 'kld.txt'), 'r') as f:
            reader = csv.reader(f)
            headers = reader.next()  # store headers
            kl_f_by_f = {int(row[0]): np.array(row[1:], dtype=np.float32) for row in reader}

        with open(join(run_dir, 'metrics', 'ig.txt'), 'r') as f:
            reader = csv.reader(f)
            headers = reader.next()  # store headers
            ig_f_by_f = {int(row[0]): np.array(row[1:], dtype=np.float32) for row in reader}

        # make a list with all the frames of this run marked as 'keep'
        # each row of `keep_this_run` is a list of four entries [num_run, start_frame, end_frame, subsequence_class]
        keep_this_run = [l for l in subsequences if l[0] == '{0}'.format(r) and l[3] == 'k']
        ranges = [range(int(r[1]), int(r[2])+1) for r in keep_this_run]
        frames_keep_this_run = [rr for r in ranges for rr in r]
        frames_keep_this_run = [f-1 for f in frames_keep_this_run]  # from matlab to python indexing (0-based)
        frames_keep_this_run = set(frames_keep_this_run)

        # keep only metrics for frames in the selected list (these are frames marked as attentive)
        filtered_cc = np.array([cc_values for key, cc_values in cc_f_by_f.iteritems() if key in frames_keep_this_run])
        filtered_kl = np.array([kld_values for key, kld_values in kl_f_by_f.iteritems() if key in frames_keep_this_run])
        filtered_ig = np.array([ig_values for key, ig_values in ig_f_by_f.iteritems() if key in frames_keep_this_run])

        mean_cc_attentive_this_run = np.nanmean(filtered_cc, axis=0)
        mean_kl_attentive_this_run = np.nanmean(filtered_kl, axis=0)
        mean_ig_attentive_this_run = np.nanmean(filtered_ig, axis=0)

        cc_all.append(mean_cc_attentive_this_run)
        kl_all.append(mean_kl_attentive_this_run)
        ig_all.append(mean_ig_attentive_this_run)

    # at the end, average results over all test runs
    mean_all_cc_attentive = np.mean(np.array(cc_all), axis=0)
    mean_all_kl_attentive = np.mean(np.array(kl_all), axis=0)
    mean_all_ig_attentive = np.mean(np.array(ig_all), axis=0)

    print('CC')
    print(mean_all_cc_attentive)
    print('')
    print('KLD')
    print(mean_all_kl_attentive)
    print('')
    print('IG')
    print(mean_all_ig_attentive)
