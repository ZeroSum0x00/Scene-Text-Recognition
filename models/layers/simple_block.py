import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from . import get_activation_from_name, get_normalizer_from_name


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=[3, 3],
                 strides=[1, 1],
                 padding='same',
                 groups=1,
                 use_bias=True,
                 activation="relu",
                 normalizer='batch-norm',
                 *args, **kwargs):
        super(ConvolutionBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        self.conv = Conv2D(filters=self.filters,
                           kernel_size=self.kernel_size,
                           strides=self.strides,
                           padding=self.padding,
                           groups=self.groups,
                           use_bias=self.use_bias)

        if self.normalizer:
            self.norm = get_normalizer_from_name(self.normalizer)

        if self.activation:
            self.activ = get_activation_from_name(self.activation)


    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        if hasattr(self, 'norm'):
            x = self.norm(x, training=training)
        if hasattr(self, 'activ'):
            x = self.activ(x, training=training)
        return x