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


def residual_block(inputs, filters, kernel_size=3, strides=1, padding="SAME"):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(x)

    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    y = BatchNormalization()(y)
    return add([x, y])


def ResUnet_FeatureExtractor(num_filters, out_dims, input_shape=(32, 400, 3)):
    f0, f1, f2, f3 = num_filters
    img_input = Input(shape=input_shape)
    
    x11 = Sequential([
        Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
    ])(img_input)
    x12 = Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(img_input)
    x1 = add([x11, x12])

    x2 = residual_block(x1, f1, 3, 2)

    x3 = residual_block(x2, f2, 3, 2)

    x4 = residual_block(x3, f3, 3, 2)
    x4 = Conv2DTranspose(filters=f3, kernel_size=(2, 2), strides=(2, 2), padding="VALID")(x4)

    x5 = concatenate([x4, x3], axis=-1)

    x6 = residual_block(x5, f2, 3, 1)
    x6 = Conv2DTranspose(filters=f2, kernel_size=(2, 2), strides=(2, 2), padding="VALID")(x6)

    x7 = concatenate([x6, x2], axis=-1)

    x8 = residual_block(x7, f1, 3, 1)
    x8 = Conv2DTranspose(filters=f1, kernel_size=(2, 2), strides=(2, 2), padding="VALID")(x8)

    x9 = concatenate([x8, x1], axis=-1)

    x10 = residual_block(x9, f0, 3, 1)

    x = Conv2D(filters=out_dims, 
               kernel_size=(1, 1), 
               strides=(1, 1), 
               padding='valid')(x10)
    x = Activation('sigmoid')(x)
    model = Model(inputs=img_input, outputs=x, name='NestedUNet')
    return model