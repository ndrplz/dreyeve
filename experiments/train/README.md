# experiments/train
Python project containing the code to train DreyeveNet.

* [`batch_generators.py`](batch_generators.py) holds functions to load 
and yield batches, and some testing code;
* [`callbacks.py`](callbacks.py) holds keras callbacks;
* [`config.py`](config.py) holds some configuration variables;
* [`custom_layers.py`](custom_layers.py) holds the bilinear upsampling layer.
* [`loss_functions.py`](loss_functions.py) holds loss functions for optimization;
* [`models.py`](models.py) holds models for saliency prediction;
* [`train.py`](train.py) holds function for training;
* [`utils.py`](utils.py) holds some helpers;
