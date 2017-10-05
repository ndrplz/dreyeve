from keras.layers.core import Layer, InputSpec
import theano.tensor as T


class BilinearUpsampling(Layer):
    def __init__(self, upsampling, input_dim=None, name='', **kwargs):
        self.name = name
        self.upsampling = upsampling
        self.channels = None

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpsampling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[1]  # todo this works only in theano
        super(BilinearUpsampling, self).build(input_shape)

    def call(self, x, mask=None):
        output = T.nnet.abstract_conv.bilinear_upsampling(x, self.upsampling,
                                                          batch_size=None, num_input_channels=self.channels)
        return output

    def get_output_shape_for(self, input_shape):
        return None, input_shape[1], input_shape[2]*self.upsampling, input_shape[3]*self.upsampling