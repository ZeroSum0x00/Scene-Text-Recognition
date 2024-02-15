import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Reshape
from tensorflow.nn import relu
from .grid_sample2 import grid_sample2, gen_grid


class LocNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        bs = input_shape[0]
        self.c1 = Conv2D(16, 3, activation=relu)
        self.c2 = Conv2D(16, 3, activation=relu)
        self.d1 = Dense(514, activation=relu)

        init = tf.constant_initializer([1, 0, 0, 0, 1, 0])
        self.d2 = Dense(6, activation=tf.tanh, bias_initializer=init)
        
    def call(self, x):

        x = self.c1(x)
        x = self.c2(x)
        b = tf.cast(tf.shape(x)[0], tf.float32)
        h = tf.cast(tf.shape(x)[1], tf.float32)
        w = tf.cast(tf.shape(x)[2], tf.float32)
        c = tf.cast(tf.shape(x)[3], tf.float32)
        x = tf.reshape(x, [-1, h*w*c])
        x = self.d1(x)
        x = self.d2(x)

        return x


class SimpleSpatialTransformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.loc = LocNet()

    def call(self, x):
        theta = self.loc(x)
        grid = gen_grid(x, theta)
        x_t = grid_sample2(x, grid)
        return x_t