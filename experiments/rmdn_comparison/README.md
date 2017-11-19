# experiments/rmdn_comparison

Some code to train and predict dreyeve sequences out of the
[Recurrent Mixture Density Network](https://openreview.net/pdf?id=SJRpRfKxx).

* [`batch_generators.py`](batch_generators.py) holds code to generate 
training batches.
* [`callbacks.py`](callbacks.py) holds Keras callbacks called during training.
* [`compute_c3d_features.py`](compute_c3d_features.py) is a script that 
computes C3D encodings for a complete dreyeve sequence.

Usage: `python compute_c3d_features.py --seq <seq>`

* [`config.py`](config.py) holds some shared configuration parameters.
* [`models.py`](models.py) holds two keras models, namely the C3D encoder 
to compute features and the RMDN model.
* [`objectives.py`](objectives.py) holds the Keras backend code for the 
negative log likelyhood loss.
* [`predict_dreyeve_sequence.py`](predict_dreyeve_sequence.py) is a script 
that can be used to compute predictions all over a sequence.

Usage: `python retrieve_c3d_encodings_from_cineca.py --host <host> --user <user> --password <password>`

* [`retrieve_c3d_encodings_from_cineca.py`](retrieve_c3d_encodings_from_cineca.py) 
is a script that retrieves all c3d precomputed encodings from cineca 
server with sftp.
* [`train.py`](train.py) is the main training script.
* [`utils.py`](utils.py) holds utility functions.
