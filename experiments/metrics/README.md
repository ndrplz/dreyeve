# experiments/metrics
Code for computing metrics over predictions.

* [`aggregate_metrics.py`](aggregate_metrics.py) script computing the mean 
of metrics in different sequences. This relies on txt files precomputed 
through `compute_metrics.py`

* [`aggregate_metrics_by_scenario.py`](aggregate_metrics_by_scenario.py) 
script that merges a metric in different scenarios, and prints a mean 
for all couples (time_of_day, wheather)

* [`calc_mean_gt_for_new_gt.py`](calc_mean_gt_for_new_gt.py) script that 
finds the mean groundtruth fixation map, which is considered a baseline.

* [`compute_metrics.py`](compute_metrics.py) script for computing metrics 
over predictions of a sequence and save them to txt files.

Usage: `python compute_metrics.py --start <start_sequence> --stop <stop_sequence>`

TODO: maybe merge the three different methods in one?

* [`compute_metrics_on_attentive.py`](compute_metrics_on_attentive.py) 
script to compute metrics on attentive subsequences only. This relies 
on txt files precomputed through `compute_metrics.py`

* [`eval_semseg_by_sequence.py`](eval_semseg_by_sequence.py) script 
that proposes random samples of semantic segmentation to the user
that has to decide if is good or bad pressing 'g' or 'b' respectively.
Selected sequences (where segmentation is good) are used to test the
semantic segmentation performance on an additional ablation study where
terrible segmentations are excluded.

* [`segmentation_stats.py`](segmentation_stats.py) is a script that reports scenario 
and actions distributions in frames in which the segmentation branch works 
better than other ones.


