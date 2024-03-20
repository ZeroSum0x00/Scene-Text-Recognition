import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import add
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from models.layers import get_activation_from_name, get_normalizer_from_name


def convolution_block(x, filters, kernel_size=3, strides=1, padding="same", use_bias=False, activation='relu', normalizer='batch-norm', classes=1000):
    if isinstance(filters, int):
        f0 = f1 = filters
    else:
        f0, f1 = filters
        
    x = Conv2D(filters=f0, 
               kernel_size=kernel_size, 
               strides=strides, 
               padding=padding,
               use_bias=use_bias)(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = Conv2D(filters=f1, 
               kernel_size=kernel_size, 
               strides=strides, 
               padding=padding,
               use_bias=use_bias)(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    return x


def downsample_block(x, filters):
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = convolution_block(x, filters=filters)
    return x


def upsample_block(inputs, filters):
    x, y = inputs
    x = Conv2DTranspose(filters=x.shape[-1]//2, kernel_size=(2, 2), strides=(2, 2))(x)
    _, xh, xw, _ = x.shape
    _, yh, yw, _ = y.shape
    diffX = yw - xw
    diffY = yh - xh
    pad = tf.constant([[0,                          0,], 
                       [diffY // 2, diffY - diffY // 2], 
                       [diffX // 2, diffX - diffX // 2], 
                       [0,                           0]])
    x = tf.pad(x, pad, mode='CONSTANT', constant_values=0)
    x = concatenate([x, y])
    x = convolution_block(x, filters)
    return x


def UNet_FeatureExtractor(num_filters, out_dims, input_shape=(32, 200, 3)):
    f0, f1, f2, f3, f4 = num_filters
    img_input = Input(shape=input_shape)
    
    x1 = convolution_block(img_input, filters=f0)
    x2 = downsample_block(x1, f1)
    x3 = downsample_block(x2, f2)
    x4 = downsample_block(x3, f3)
    x5 = downsample_block(x4, f4)

    x = upsample_block([x5, x4], f3)
    x = upsample_block([x, x3], f2)
    x = upsample_block([x, x2], f1)
    x = upsample_block([x, x1], f0)
    x = Conv2D(filters=out_dims, 
               kernel_size=(1, 1), 
               strides=(1, 1), 
               padding='valid')(x)
    model = Model(inputs=img_input, outputs=x, name='UNet')
    return model