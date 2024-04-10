import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate

from .unet import convolution_block
from models.layers import get_activation_from_name, get_normalizer_from_name


class AttentionBlock(tf.keras.layers.Layer):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, filters, activation='relu', normalizer='batch-norm', *args, **kwargs):
        super(AttentionBlock, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.activation = activation
        self.normalizer = normalizer
        
    def build(self, input_shape):
        self.W_gate = Sequential([
            Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding="VALID"),
            get_normalizer_from_name(self.normalizer),
        ])
        self.W_x = Sequential([
            Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding="VALID"),
            get_normalizer_from_name(self.normalizer),
        ])
        self.psi = Sequential([
            Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding="VALID"),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name('sigmoid')
        ])
        self.activ = get_activation_from_name(self.activation)

    def call(self, inputs, training=False):
        gate, skip_connection = inputs
        g1 = self.W_gate(gate, training=training)
        x1 = self.W_x(skip_connection, training=training)
        psi = self.activ(g1 + x1, training=training)
        psi = self.psi(psi, training=training)
        x = skip_connection * psi
        return x

        
def upsample_block(inputs, filters, activation='relu', normalizer='batch-norm'):
    x = UpSampling2D(size=(2, 2))(inputs)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True)(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    return x


def AttentionUNet_FeatureExtractor(num_filters, out_dims, input_shape=(32, 400, 3), classes=1000):
    f0, f1, f2, f3, f4 = num_filters
    img_input = Input(shape=input_shape)
    
    e1 = convolution_block(img_input, filters=f0, use_bias=True)

    e2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(e1)
    e2 = convolution_block(e2, filters=f1, use_bias=True)

    e3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(e2)
    e3 = convolution_block(e3, filters=f2, use_bias=True)

    e4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(e3)
    e4 = convolution_block(e4, filters=f3, use_bias=True)

    e5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(e4)
    e5 = convolution_block(e5, filters=f4, use_bias=True)

    d5 = upsample_block(e5, f3)
    s4 = AttentionBlock(f2)([d5, e4])
    d5 = concatenate([s4, d5], axis=-1)
    d5 = convolution_block(d5, filters=f3, use_bias=True)

    d4 = upsample_block(d5, f2)
    s3 = AttentionBlock(f1)([d4, e3])
    d4 = concatenate([s3, d4], axis=-1)
    d4 = convolution_block(d4, filters=f2, use_bias=True)

    d3 = upsample_block(d4, f1)
    s2 = AttentionBlock(f0)([d3, e2])
    d3 = concatenate([s2, d3], axis=-1)
    d3 = convolution_block(d3, filters=f1, use_bias=True)

    d2 = upsample_block(d3, f0)
    s1 = AttentionBlock(f0 // 2)([d2, e1])
    d2 = concatenate([s1, d2], axis=-1)
    d2 = convolution_block(d2, filters=f0, use_bias=True)

    x = Conv2D(filters=out_dims, 
               kernel_size=(1, 1), 
               strides=(1, 1), 
               padding='valid')(d2)
    model = Model(inputs=img_input, outputs=x, name='AttentionUNet')
    return model