import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from . import get_activation_from_name, get_normalizer_from_name
from .grid_sample import grid_sample_with_mask


class LocalizationNetwork(tf.keras.Model):

    def __init__(self, num_control_points, control_activation=False, activation='relu', normalizer='batch-norm', *args, **kwargs):
        super(LocalizationNetwork, self).__init__(*args, **kwargs)
        self.num_control_points = num_control_points
        self.control_activation = control_activation
        self.activation         = activation
        self.normalizer         = normalizer
        
    def build(self, input_shape):
        self.conv = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
        ])

        self.localization_fc1 = Sequential([
            Dense(units=512),
            get_normalizer_from_name(self.normalizer),
            get_activation_from_name(self.activation),
        ])

        margin = 0.01
        sampling_num_per_side = int(self.num_control_points / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)

        if self.control_activation:
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        else:
            pass

        initial_bias = tf.constant_initializer(ctrl_points.reshape(-1))
        initial_weights = tf.initializers.zeros()
        self.localization_fc2 = Dense(units=self.num_control_points * 2, kernel_initializer=initial_weights, bias_initializer=initial_bias)

    def call(self, inputs, training=False):
        bs = tf.shape(inputs)[0]
        x = self.conv(inputs, training=training)
        x = tf.reshape(x, [bs, -1])
        x = self.localization_fc1(x, training=training)
        x = self.localization_fc2(0.1 * x, training=training)
        if self.control_activation:
            x = tf.nn.sigmoid(x)
        x = tf.reshape(x, shape=[-1, self.num_control_points, 2])
        return x


class TPS_SpatialTransformerNetworkV2(tf.keras.layers.Layer):

    def __init__(self, grid_size=(32, 64), num_control_points=20, margins=[0.05, 0.05], *args, **kwargs):
        super(TPS_SpatialTransformerNetworkV2, self).__init__(*args, **kwargs)
        self.grid_size          = grid_size
        self.num_control_points = num_control_points
        self.margins            = margins

    def build(self, input_shape):
        bs = input_shape[0]
        image_size = input_shape[1:-1]
        target_control_points = self.build_output_control_points()
        target_control_points = tf.convert_to_tensor(target_control_points, dtype=tf.float32)

        N = self.num_control_points

        # create padded kernel matrix
        forward_kernel = np.zeros((N + 3, N + 3))
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N] = target_control_partial_repr
        forward_kernel[:N, -3] = 1
        forward_kernel[-3, :N] = 1
        forward_kernel[:N, -2:] = target_control_points
        forward_kernel[-2:, :N] = np.transpose(target_control_points, [1, 0])
        inverse_kernel = np.linalg.inv(forward_kernel)
        inverse_kernel = tf.convert_to_tensor(inverse_kernel, dtype=tf.float32)
        self.inverse_kernel = tf.Variable(
            initial_value=inverse_kernel, trainable=True, name=f'TPS_SpatialTransformerNetworkV2/inverse_kernel'
        )

        # create target cordinate matrix
        target_coordinate = list(itertools.product(range(image_size[0]), range(image_size[1])))
        target_coordinate = np.array(target_coordinate)
        Y, X = np.split(target_coordinate, indices_or_sections=2, axis=-1)
        Y = Y / (image_size[0] - 1)
        X = X / (image_size[1] - 1)
        target_coordinate = np.concatenate([X, Y], axis=-1)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
        ones_matrix = np.ones((image_size[0] * image_size[1], 1))
        target_coordinate_repr = np.concatenate([target_coordinate_partial_repr, ones_matrix, target_coordinate], axis=-1)
        target_coordinate_repr = tf.convert_to_tensor(target_coordinate_repr, dtype=tf.float32)
        self.target_coordinate_repr = tf.Variable(
            initial_value=target_coordinate_repr, trainable=True, name=f'TPS_SpatialTransformerNetworkV2/target_coordinate_repr'
        )

        # create padding matrix
        padding_matrix = np.zeros((1, 3, 2))
        padding_matrix = tf.convert_to_tensor(padding_matrix, dtype=tf.float32)
        self.padding_matrix = tf.Variable(
            initial_value=padding_matrix, trainable=True, name=f'TPS_SpatialTransformerNetworkV2/padding_matrix'
        )
        
        self.localization_network = LocalizationNetwork(self.num_control_points, control_activation=True)

    def build_output_control_points(self):
        margin_x, margin_y = self.margins
        num_ctrl_pts_per_side = self.num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        # ctrl_pts_top = ctrl_pts_top[1:-1,:]
        # ctrl_pts_bottom = ctrl_pts_bottom[1:-1,:]
        output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return output_ctrl_pts_arr

    def compute_partial_repr(self, input_points, control_points):
        N = input_points.shape[0]
        M = control_points.shape[0]
        input_points = np.expand_dims(input_points, axis=1)
        control_points = np.expand_dims(control_points, axis=0)

        pairwise_diff = input_points - control_points
        # original implementation, very slow
        # pairwise_dist = tf.reduce_sum(pairwise_diff ** 2, axis=2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * np.log(pairwise_dist, where=pairwise_dist > 0)
        return repr_matrix

    def call(self, inputs, training=False):
        stn_input = tf.image.resize(inputs, size=self.grid_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        source_control_points = self.localization_network(stn_input, training=training)
        bs = tf.shape(inputs)[0]
        h  = tf.shape(inputs)[1]
        w  = tf.shape(inputs)[2]
        padding_value = tf.repeat(self.padding_matrix, bs, axis=0)
        Y = tf.concat([source_control_points, padding_value], axis=1)
        mapping_matrix = tf.matmul(self.inverse_kernel, Y)
        source_coordinate = tf.matmul(self.target_coordinate_repr, mapping_matrix)
        grid = tf.reshape(source_coordinate, shape=[bs, h, w, -1])
        grid = tf.clip_by_value(grid, clip_value_min=0, clip_value_max=1)
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample_with_mask(inputs, grid=grid, canvas=None, mode="bilinear", padding_mode="zeros", align_corners=True)
        return output_maps