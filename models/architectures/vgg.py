import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from models.layers import get_activation_from_name, get_normalizer_from_name


def VGG_FeatureExtractor(num_filters, input_shape=(32, 200, 3), activation='relu', normalizer='batch-norm', classes=1000, drop_rate=0.35):
    f0, f1, f2, f3 = num_filters
    img_input = Input(shape=input_shape)
    
    x = Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(img_input)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = SpatialDropout2D(drop_rate)(x)

    x = Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = SpatialDropout2D(drop_rate)(x)
    
    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_activation_from_name(activation)(x)
    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = SpatialDropout2D(drop_rate)(x)

    x = Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = SpatialDropout2D(drop_rate)(x)
    x = Conv2D(filters=f3, kernel_size=(2, 2), strides=(1, 1), padding='valid', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = get_activation_from_name(activation)(x)
    x = SpatialDropout2D(drop_rate)(x)
    model = Model(inputs=img_input, outputs=x, name='VGG')
    return model
