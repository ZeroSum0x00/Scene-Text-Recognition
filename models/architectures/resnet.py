import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import add
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from models.layers import get_activation_from_name, get_normalizer_from_name


def BasicBlock(input_tensor, filters, kernel_size=3, downsaple=False, activation='relu', normalizer='batch-norm'):
    filter1, filter2 = filters
    shortcut = input_tensor
    strides = 1
    x = Conv2D(filters=filter1, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(input_tensor)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)

    x = Conv2D(filters=filter2, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)

    if downsaple:
        shortcut = Conv2D(filters=filter2, kernel_size=(1, 1), strides=strides, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(shortcut)
        shortcut = get_normalizer_from_name(normalizer)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet_FeatureExtractor(input_shape=(32, 200, 1), num_blocks=[1, 2, 5, 3], activation='relu', normalizer='batch-norm'):
    img_input = Input(shape=input_shape)
    
    # Block conv1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(img_input)
    x = get_normalizer_from_name(normalizer)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = SpatialDropout2D(0.35)(x)

    # Block conv2_x
    for i in range(num_blocks[0]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [128, 128], 3, downsaple)
        
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = SpatialDropout2D(0.35)(x)

    # Block conv3_x
    for i in range(num_blocks[1]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [256, 256], 3, downsaple)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = SpatialDropout2D(0.35)(x)

    # Block conv4_x
    for i in range(num_blocks[2]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [512, 512], 3, downsaple)
        
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)

    # Block conv5_x
    for i in range(num_blocks[3]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [512, 512], 3, downsaple)
        
    x = Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = SpatialDropout2D(0.35)(x)
 
    model = Model(inputs=img_input, outputs=x, name='ResNetX')
    return model
