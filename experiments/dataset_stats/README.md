### experiments/dataset_stats

This folder holds code to mine the DR(eye)VE dataset for statistics.

* [`acting_subs_by_scenario.py`](acting_subs_by_scenario.py) computes 
an histogram holding the percentage of acting frames in each scenario.
* [`log_variance_stats.py`](log_variance_stats.py) is a script
that computes variance stats in groundtruth attentional maps and logs
them on a file for successive visualization.
* [`plot_variance_stats.py`](plot_variance_stats.py) is a script that
reads the file and visualizes plots.
* [`stats_utils.py`](stats_utils.py) holds some statistical and
utility functions.
