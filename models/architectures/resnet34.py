import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name


def BasicBlock(input_tensor, filters, kernel_size=3, strides=(1, 1), downsaple=False, activation='relu', normalizer='batch-norm'):
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(input_tensor)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)

    if downsaple:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='VALID', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(input_tensor)
        shortcut = BatchNormalization()(shortcut)
        shortcut = get_normalizer_from_name(normalizer)(shortcut)

    else:
        shortcut = input_tensor

    x = add([x, shortcut])
    x = get_activation_from_name(activation)(x)
    return x


def ResNet34(num_blocks, input_shape=(32, 128, 3), activation='relu', normalizer='batch-norm', classes=1000):
    img_input = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(img_input)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    
    for i in range(num_blocks[0]):
        downsaple = True if i == 0 else False
        strides = 2 if i == 0 else 1
        x = BasicBlock(x, 32, 3, strides, downsaple)

    for i in range(num_blocks[1]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, 64, 3, 1, downsaple)

    for i in range(num_blocks[2]):
        downsaple = True if i == 0 else False
        strides = 2 if i == 0 else 1
        x = BasicBlock(x, 128, 3, strides, downsaple)

    for i in range(num_blocks[3]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, 256, 3, 1, downsaple)

    for i in range(num_blocks[4]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, 512, 3, 1, downsaple)
    model = Model(img_input, x)
    return model