import tensorflow as tf
import numpy as np 


def gen_grid(im, theta):
    
    def _target_grid(height, width):
        height = tf.cast(height, tf.int32)
        width = tf.cast(width, tf.int32)
        x_t, y_t = tf.meshgrid(
            tf.linspace(-1.0, 1.0, width),
            tf.linspace(-1.0, 1.0, height)
        )
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])
        ones = tf.ones_like(x_t_flat)
    
        grid = tf.stack([x_t_flat, y_t_flat, ones])
        return grid
        
    b = tf.cast(tf.shape(im)[0], tf.float32)
    h = tf.cast(tf.shape(im)[1], tf.float32)
    w = tf.cast(tf.shape(im)[2], tf.float32)
    c = tf.cast(tf.shape(im)[3], tf.float32)
    # b, h, w, c = im.shape
    theta = tf.reshape(theta, (-1, 2, 3))

    grid = _target_grid(h, w)
    grid = tf.expand_dims(grid, 0)
    grid = tf.tile(grid, [b, 1, 1])

    T = theta @ grid
    T = tf.reshape(T, [-1, 2, h, w])
    return tf.transpose(T, [0, 2, 3, 1])


def grid_sample2(im, grid):
    b = tf.cast(tf.shape(im)[0], tf.float32)
    h = tf.cast(tf.shape(im)[1], tf.float32)
    w = tf.cast(tf.shape(im)[2], tf.float32)
    c = tf.cast(tf.shape(im)[3], tf.float32)
    # b, h, w, c = im.shape

    h_f, w_f = tf.cast(h, tf.float32), tf.cast(w, tf.float32)

    grid_x = grid[:, :, :, 0]
    grid_y = grid[:, :, :, 1]

    grid_x = (grid_x + 1) * w_f / 2
    grid_y = (grid_y + 1) * h_f / 2

    grid_x0 = tf.floor(grid_x)
    grid_x1 = grid_x0 + 1
    grid_y0 = tf.floor(grid_y)
    grid_y1 = grid_y0 + 1

    grid_x0 = tf.clip_by_value(grid_x0, 0, w_f - 2)
    grid_x1 = tf.clip_by_value(grid_x1, 1, w_f - 1)
    grid_y0 = tf.clip_by_value(grid_y0, 0, h_f - 2)
    grid_y1 = tf.clip_by_value(grid_y1, 1, h_f - 1)

    b_index = tf.reshape(tf.range(b), [b, 1, 1])
    b_index = tf.tile(b_index, [1, h, w])
    b_index = tf.cast(b_index, tf.int32)

    grid_x0_i = tf.cast(grid_x0, tf.int32)
    grid_x1_i = tf.cast(grid_x1, tf.int32)
    grid_y0_i = tf.cast(grid_y0, tf.int32)
    grid_y1_i = tf.cast(grid_y1, tf.int32)

    Q00 = tf.gather_nd(im, tf.stack([b_index, grid_y0_i, grid_x0_i], -1))
    Q01 = tf.gather_nd(im, tf.stack([b_index, grid_y1_i, grid_x0_i], -1))
    Q10 = tf.gather_nd(im, tf.stack([b_index, grid_y0_i, grid_x1_i], -1))
    Q11 = tf.gather_nd(im, tf.stack([b_index, grid_y1_i, grid_x1_i], -1))
    Q = tf.stack([Q00, Q01, Q10, Q11], -1)
    Q = tf.reshape(Q, [b, h, w, c, 2, 2])

    Wx = tf.stack([grid_x1 - grid_x, grid_x - grid_x0], -1)
    Wx = tf.reshape(Wx, [b, h, w, 1, 1, 2])
    Wx = tf.tile(Wx, [1, 1, 1, c, 1, 1])

    Wy = tf.stack([grid_y1 - grid_y, grid_y - grid_y0], -1)
    Wy = tf.reshape(Wy, [b, h, w, 1, 2, 1])
    Wy = tf.tile(Wy, [1, 1, 1, c, 1, 1])

    Wd = (grid_x1 - grid_x0) * (grid_y1 - grid_y0)
    Wd = tf.expand_dims(Wd, -1)

    im_t = Wx @ Q @ Wy
    im_t = tf.reshape(im_t, [b, h, w, c])
    im_t = im_t / Wd
    return im_t