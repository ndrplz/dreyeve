# experiments/visualization
Some helper code for paper figures.

* [`prepare_showcase_blend.py`](`prepare_showcase_blend.py`)
script for preparing showcase videos.
* [`utils.py`](`utils.py`) holds a function for blending maps and frames.
* [`visualize_comparison`](`visualize_comparison`) script that selects some figures where our model performs better
than competitors and saves them in a separate folder for a figure.
* [`visualize_comparison_ablation.py`](`visualize_comparison_ablation.py`)
as above, but compares the final predictions with the ones of
single branches.
* [`visualize_comparison_segmentation.py`](`visualize_comparison_segmentation.py`)
as above, but extracts frames in which segmentation helps.
* [`visualize_predictions`](`visualize_predictions`) script that visualizes predictions over random samples,
in an infinite loop.
