import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D


class VGG(tf.keras.layers.Layer):
    def __init__(self, num_filters, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)
        self.num_filters = num_filters
    
    def build(self, input_shape):
        f0, f1, f2, f3 = self.num_filters
        self.block0 = Sequential([
                        Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                        SpatialDropout2D(0.35)
        ])
        self.block1 = Sequential([
                        Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                        SpatialDropout2D(0.35)
        ])
        self.block2 = Sequential([
                        Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
                        SpatialDropout2D(0.35)
        ])
        self.block3 = Sequential([
                        Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False),
                        BatchNormalization(),
                        Activation('relu'),
                        Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False),
                        BatchNormalization(),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
                        SpatialDropout2D(0.35),
                        Conv2D(filters=f3, kernel_size=(2, 2), strides=(1, 1), padding='valid'),
                        Activation('relu'),
                        SpatialDropout2D(0.35)
        ])
    def call(self, inputs, training=False):
        x = self.block0(inputs, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        return x
