import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from models.layers import get_activation_from_name, get_normalizer_from_name


def ShallowCNN(filters=512, input_shape=(32, 128, 3), activation='relu', normalizer='batch-norm'):
    img_input = Input(shape=input_shape)
    x = Conv2D(filters=filters // 2, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False)(img_input)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False)(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    model = Model(img_input, x)
    return model