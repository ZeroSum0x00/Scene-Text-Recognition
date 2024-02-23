import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import concatenate
from . import get_activation_from_name, get_normalizer_from_name
from .grid_sample import grid_sample_with_mask


class GridGenerator(tf.keras.layers.Layer):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size, *args, **kwargs):
        super(GridGenerator, self).__init__(*args, **kwargs)
        self.F = F
        self.I_r_size = I_r_size
        self.eps = 1e-6

    def build(self, input_shape):
        bs = input_shape[0]
        
        """ Return coordinates of fiducial points in I_r; C """
        I_r_height, I_r_width = self.I_r_size
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(self.F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(self.F / 2))
        ctrl_pts_y_bottom = np.ones(int(self.F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((self.F, self.F), dtype=float)  # F x F
        for i in range(0, self.F):
            for j in range(i, self.F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((self.F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, self.F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        self.inv_delta_C = np.linalg.inv(delta_C)
        self.inv_delta_C = np.expand_dims(self.inv_delta_C, axis=0)
        self.inv_delta_C = tf.convert_to_tensor(self.inv_delta_C, dtype=tf.float32)
        self.inv_delta_C = tf.Variable(
            initial_value=self.inv_delta_C, trainable=True, name=f'grid_generator/inv_delta_C'
        )
        
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        ).reshape([-1, 2])
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, self.F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        self.P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        self.P_hat = np.expand_dims(self.P_hat, axis=0)
        self.P_hat = tf.convert_to_tensor(self.P_hat, dtype=tf.float32)
        self.P_hat = tf.Variable(
            initial_value=self.P_hat, 
            trainable=True, 
            name=f'grid_generator/P_hat'
        )

        self.batch_zeros = self.add_weight(
            f'grid_generator/batch_zeros',
            shape       = (bs, 3, 2),
            initializer = tf.initializers.zeros(),
            trainable   = True
        )

    def call(self, inputs, training=False):
        bs = tf.shape(inputs)[0]
        batch_inv_delta_C = tf.repeat(self.inv_delta_C, bs, axis=0)
        batch_P_hat       = tf.repeat(self.P_hat, bs, axis=0)
        batch_C_prime_with_zeros = concatenate([inputs, self.batch_zeros], axis=1)
        batch_T = tf.matmul(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = tf.matmul(batch_P_hat, batch_T)
        return batch_P_prime


class LocalizationNetwork(tf.keras.layers.Layer):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, activation='relu', normalizer='batch-norm', *args, **kwargs):
        super(LocalizationNetwork, self).__init__(*args, **kwargs)
        self.F          = F
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        bs = input_shape[0]
        self.conv = Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            GlobalAveragePooling2D(),
        ])
        self.localization_fc1 = Sequential([
            Dense(units=256),
            get_activation_from_name(self.activation),
        ])

        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(self.F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(self.F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(self.F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).reshape(-1)
        initial_bias = tf.constant_initializer(initial_bias)
        initial_weights = tf.initializers.zeros()
        self.localization_fc2 = Dense(units=self.F * 2, kernel_initializer=initial_weights, bias_initializer=initial_bias)
        
    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        x = self.localization_fc1(x, training=training)
        x = self.localization_fc2(x, training=training)
        x = tf.reshape(x, shape=[-1, self.F, 2])
        return x


class TPS_SpatialTransformerNetwork(tf.keras.layers.Layer):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F=20, *args, **kwargs):
        super(TPS_SpatialTransformerNetwork, self).__init__(*args, **kwargs)
        self.F = F

    def build(self, input_shape):
        self.I_r_size = input_shape[1:-1]
        self.localization_network = LocalizationNetwork(self.F)
        self.grid_generator = GridGenerator(self.F, self.I_r_size)

    def call(self, inputs, training=False):
        batch_C_prime = self.localization_network(inputs, training=training)
        build_P_prime = self.grid_generator(batch_C_prime, training=training)
        build_P_prime_reshape = tf.reshape(build_P_prime, shape=[-1, self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = grid_sample_with_mask(inputs, grid=build_P_prime_reshape, canvas=None, mode="bilinear", padding_mode="border", align_corners=True)
        return batch_I_r