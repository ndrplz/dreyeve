# experiments/rmdn_comparison

Some code to train and predict dreyeve sequences out of the
[Recurrent Mixture Density Network](https://openreview.net/pdf?id=SJRpRfKxx).

* `models.py` holds two keras models, namely the C3D encoder to compute features and
the RMDN model.
* `compute_c3d_features.py` is a script that computes C3D encodings for a complete
dreyeve sequence.

Usage: `python compute_c3d_features.py --seq <seq>`