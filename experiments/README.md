# experiments
This folder holds python code used for the experimental section.

* [`actions`](actions) holds code for predicting actions given the predicted map
(supplementary material)
* [`assessment`](assessment) holds code to run the visual
assessment experiment (sec 5.4 of the paper)
* [`dataset_stats`](dataset_stats) holds code to measure some 
data-related statistics.
* [`metrics`](metrics) holds code to compute metrics from precomputed predictions.
* [`mlnet_comparison`](mlnet_comparison) holds code to train and predict a sequence from [Multi Level Network](https://github.com/marcellacornia/mlnet) from Cornia et al.
* [`predict_on_cineca`](predict_on_cineca) holds code to predict sequences with the dreyevenet model.
* [`rmdn_comparison`](rmdn_comparison) holds code to extract C3D features, train and predict using
the [Recurrend Mixture Density Network](https://openreview.net/pdf?id=SJRpRfKxx) from Bazzani et al.
* [`train`](train) holds code to train the dreyevenet model.
* [`visualization`](visualization) holds code for visualizing some predictions (mainly used for paper figures)

**All python code has been developed and tested with Keras 1 and using Theano as backend.**
