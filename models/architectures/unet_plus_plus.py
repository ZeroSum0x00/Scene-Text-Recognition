import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import add
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from .unet import convolution_block


def NestedUNet_FeatureExtractor(num_filters, out_dims, input_shape=(32, 400, 3)):
    f0, f1, f2, f3, f4 = num_filters
    img_input = Input(shape=input_shape)
    
    x0_0 = convolution_block(img_input, filters=f0, use_bias=True)
    x1_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x0_0)
    x1_0 = convolution_block(x1_0, filters=f1, use_bias=True)
    x0_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1_0)
    x0_1 = concatenate([x0_0, x0_1], axis=-1)
    x0_1 = convolution_block(x0_1, filters=f0, use_bias=True)

    x2_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1_0)
    x2_0 = convolution_block(x2_0, filters=f2, use_bias=True)
    x1_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x2_0)
    x1_1 = concatenate([x1_0, x1_1], axis=-1)
    x1_1 = convolution_block(x1_1, filters=f1, use_bias=True)
    x0_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1_1)
    x0_2 = concatenate([x0_0, x0_1, x0_2], axis=-1)
    x0_2 = convolution_block(x0_2, filters=f0, use_bias=True)

    x3_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2_0)
    x3_0 = convolution_block(x3_0, filters=f3, use_bias=True)
    x2_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x3_0)
    x2_1 = concatenate([x2_0, x2_1], axis=-1)
    x2_1 = convolution_block(x2_1, filters=f2, use_bias=True)
    x1_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x2_1)
    x1_2 = concatenate([x1_0, x1_1, x1_2], axis=-1)
    x1_2 = convolution_block(x1_2, filters=f1, use_bias=True)
    x0_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1_2)
    x0_3 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    x0_3 = convolution_block(x0_3, filters=f0, use_bias=True)

    x4_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3_0)
    x4_0 = convolution_block(x4_0, filters=f4, use_bias=True)
    x3_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x4_0)
    x3_1 = concatenate([x3_0, x3_1], axis=-1)
    x3_1 = convolution_block(x3_1, filters=f3, use_bias=True)
    x2_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x3_1)
    x2_2 = concatenate([x2_0, x2_1, x2_2], axis=-1)
    x2_2 = convolution_block(x2_2, filters=f2, use_bias=True)
    x1_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x2_2)
    x1_3 = concatenate([x1_0, x1_1, x1_2, x1_3], axis=-1)
    x1_3 = convolution_block(x1_3, filters=f1, use_bias=True)
    x0_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1_3)
    x0_4 = concatenate([x0_0, x0_1, x0_2, x0_3, x0_4], axis=-1)
    x0_4 = convolution_block(x0_4, filters=f0, use_bias=True)

    x = Conv2D(filters=out_dims, 
               kernel_size=(1, 1), 
               strides=(1, 1), 
               padding='valid')(x0_4)
    model = Model(inputs=img_input, outputs=x, name='NestedUNet')
    return model