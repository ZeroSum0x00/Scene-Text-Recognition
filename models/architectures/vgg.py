import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D


def VGG_FeatureExtractor(num_filters, input_shape=(32, 200, 3)):
    f0, f1, f2, f3 = num_filters
    img_input = Input(shape=input_shape)
    
    x = Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding='same')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = SpatialDropout2D(0.35)(x)

    x = Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = SpatialDropout2D(0.35)(x)
    
    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = SpatialDropout2D(0.35)(x)

    x = Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = SpatialDropout2D(0.35)(x)
    x = Conv2D(filters=f3, kernel_size=(2, 2), strides=(1, 1), padding='valid')(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.35)(x)
    model = Model(inputs=img_input, outputs=x, name='VGG')
    return model
