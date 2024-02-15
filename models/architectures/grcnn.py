import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import add


def GRCL_unit(inputs):
    wgf_u, wgr_x, wf_u, wr_x = inputs
    G_first_term = BatchNormalization()(wgf_u)
    G_second_term = BatchNormalization()(wgr_x)
    G = tf.math.sigmoid(G_first_term + G_second_term)
    x_first_term = BatchNormalization()(wf_u)
    x_second_term = BatchNormalization()(BatchNormalization()(wr_x) * G)
    x = tf.nn.relu(x_first_term + x_second_term)
    return x


def GRCL(inputs, filters, kernel_size, num_iteration):
    wgf_u = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(inputs)
    wf_u  = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False)(inputs)
    x     = Activation('relu')(BatchNormalization()(wf_u))

    for i in range(num_iteration):
        next_wgr_x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
        next_wr_x  = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False)(x)
        x          = GRCL_unit([wgf_u, next_wgr_x, wf_u, next_wr_x])
    return x


def GRCNN_FeatureExtractor(input_shape=(32, 100, 1)):
    """ 
        FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) 
    """
    
    img_input = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = GRCL(x, 64, 3, 5)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = GRCL(x, 128, 3, 5)
    x = ZeroPadding2D(padding=[(0, 0), (0, 1)])(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = GRCL(x, 256, 3, 5)
    x = ZeroPadding2D(padding=[(0, 0), (0, 1)])(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    model = Model(inputs=img_input, outputs=x, name='RCNN-Feature-Extractor')
    return model
