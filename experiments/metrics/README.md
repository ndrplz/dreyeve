# experiments/metrics
Code for computing metrics over predictions.

* `compute_metrics.py` script for computing metrics over predictions of a sequence
and save them to txt files.

Usage: `python compute_metrics.py --start <start_sequence> --stop <stop_sequence>`

TODO: maybe merge the three different methods in one?

* `aggregate_metrics.py` script computing the mean of metrics in different sequences.
This relies on txt files precomputed through `compute_metrics.py`

* `compute_metrics_on_attentive.py` script to compute metrics on attentive subsequences only.
This relies on txt files precomputed through `compute_metrics.py`
