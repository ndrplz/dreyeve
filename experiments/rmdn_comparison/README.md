# experiments/rmdn_comparison

Some code to train and predict dreyeve sequences out of the
[Recurrent Mixture Density Network](https://openreview.net/pdf?id=SJRpRfKxx).

* `models.py` holds two keras models, namely the C3D encoder to compute features and
the RMDN model.
* `callbacks.py` holds Keras callbacks called during training.
* `batch_generators.py` holds code to generate training batches.
* `config.py` holds some shared configuration parameters.
* `objectives.py` holds the Keras backend code for the negative log likelyhood loss.
* `compute_c3d_features.py` is a script that computes C3D encodings for a complete
dreyeve sequence.

Usage: `python compute_c3d_features.py --seq <seq>`

* `retrieve_c3d_encodings_from_cineca.py` is a script that retrieves all c3d
precomputed encodings from cineca server with sftp.

Usage: `python retrieve_c3d_encodings_from_cineca.py --host <host> --user <user> --password <password>`

* `predict_dreyeve_sequence.py` is a script that can be used to compute predictions
all over a sequence.

