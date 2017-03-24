from keras.layers.core import Layer, InputSpec
from keras import constraints, regularizers, initializations, activations
import keras.backend as K
import theano.tensor as T


class EltWiseProduct(Layer):
    def __init__(self, downsampling_factor=10, init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):

        self.downsampling_factor = downsampling_factor
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

        self.input_spec = [InputSpec(ndim=4)]
        super(EltWiseProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init([s // self.downsampling_factor for s in input_shape[2:]])

        self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        output = x*T.nnet.abstract_conv.bilinear_upsampling(K.expand_dims(K.expand_dims(1 + self.W, 0), 0), self.downsampling_factor, 1, 1)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.input_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim,
                  'downsampling_factor': self.downsampling_factor}
        base_config = super(EltWiseProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))