import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from models.layers import get_activation_from_name, get_normalizer_from_name


def residual_block(inputs, filters, kernel_size=3, strides=1, padding="SAME", activation='relu', normalizer='batch-norm'):
    x = get_normalizer_from_name(normalizer)(inputs)
    x = get_activation_from_name(activation)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(x)

    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    y = get_normalizer_from_name(normalizer)(y)
    return add([x, y])


def ResUnet_FeatureExtractor(num_filters, out_dims, input_shape=(32, 400, 3), activation='relu', normalizer='batch-norm'):
    f0, f1, f2, f3 = num_filters
    img_input = Input(shape=input_shape)
    
    x11 = Sequential([
        Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
        get_normalizer_from_name(normalizer),
        get_activation_from_name(activation),
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
    x = get_activation_from_name('sigmoid')(x)
    model = Model(inputs=img_input, outputs=x, name='NestedUNet')
    return model