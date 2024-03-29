import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D


def gather(input, y, x, b, h, w, c, padding_mode):
    # Slow!
    # return tf.gather_nd(params=input, indices=tf.cast(tf.concat([y, x], axis=-1), dtype=tf.int32), batch_dims=1)

    if padding_mode == 'zeros':
        w_padded = w + 2
        h_padded = h + 2
        linear_coordinates = tf.cast(y * w_padded + x, dtype=tf.int32)
        linear_coordinates = tf.reshape(linear_coordinates, shape=(b, h, w))
        input = tf.reshape(input, shape=(b, h_padded * w_padded, c))
    else:
        linear_coordinates = tf.cast(y * w + x, dtype=tf.int32)
        linear_coordinates = tf.reshape(linear_coordinates, shape=(b, h, w))
        input = tf.reshape(input, shape=(b, h * w, c))

    out = tf.gather(params=input, indices=linear_coordinates, batch_dims=1)
    return out

    
def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    assert mode in ('bilinear', 'nearest')
    assert padding_mode in ('border', 'zeros', 'reflection')

    # (b, w, 2, h) --> (b, h, w, 2)
    # grid = tf.transpose(grid, perm=(0, 3, 1, 2))

    b = tf.cast(tf.shape(input)[0], tf.float32)
    h = tf.cast(tf.shape(input)[1], tf.float32)
    w = tf.cast(tf.shape(input)[2], tf.float32)
    c = tf.cast(tf.shape(input)[3], tf.float32)
    # b, h, w, c = tf.cast(tf.shape(input), tf.float32)

    def process_coord(grid, w_h):
        if align_corners:
            pixs = (grid + 1) * (0.5 * (w_h - 1))
        else:
            pixs = (grid + 1) * (0.5 * w_h) - 0.5

        if padding_mode == 'border':
            pixs = tf.clip_by_value(pixs, 0, w_h - 1)
        elif padding_mode == 'zeros':
            pixs = tf.clip_by_value(pixs, -1, w_h) + 1
        elif padding_mode == 'reflection':
            if align_corners:
                #                                      Avoid % 0
                pixs = (w_h - 1) - tf.abs(pixs % (2*tf.maximum(w_h - 1, 1)) - (w_h - 1))
            else:
                pixs = w_h - tf.abs((pixs + 0.5) % (2*w_h) - w_h) - 0.5
                pixs = tf.clip_by_value(pixs, 0, w_h - 1)
        return pixs

    # Somehow it's faster to process them separately
    grid_x, grid_y = tf.split(grid, num_or_size_splits=2, axis=-1)
    x = process_coord(grid_x, w)
    y = process_coord(grid_y, h)

    # As opposed to:
    # xy = process_coord(grid, tf.stack([w, h]))
    # x, y = tf.split(xy, num_or_size_splits=2, axis=-1)

    if padding_mode == 'zeros':
        input = ZeroPadding2D(padding=(1, 1))(input)

    if mode == 'bilinear':
        x0 = tf.math.floor(x)
        y0 = tf.math.floor(y)
        x1 = tf.math.ceil(x)
        y1 = tf.math.ceil(y)

        '''
        bilinear interpolation
            image[x, y]  =  (1 - dy) * (1 - dx) * image[y0, x0] +
                               dy    * (1 - dx) * image[y1, x0] +
                               dy    *    dx    * image[y1, x1] +
                            (1 - dy) *    dx    * image[y0, x1]
        '''
        dx = x - x0
        dy = y - y0
        oneminus_dx = 1 - dx
        oneminus_dy = 1 - dy
        w_y0_x0 = oneminus_dy * oneminus_dx
        w_y1_x0 = dy * oneminus_dx
        w_y1_x1 = dy * dx
        w_y0_x1 = oneminus_dy * dx

        v_y0_x0 = gather(input, y0, x0, b, h, w, c, padding_mode)
        v_y1_x0 = gather(input, y1, x0, b, h, w, c, padding_mode)
        v_y1_x1 = gather(input, y1, x1, b, h, w, c, padding_mode)
        v_y0_x1 = gather(input, y0, x1, b, h, w, c, padding_mode)
        return w_y0_x0 * v_y0_x0 + w_y1_x0 * v_y1_x0 + w_y1_x1 * v_y1_x1 + w_y0_x1 * v_y0_x1

    elif mode == 'nearest':
        # NB: Rounding operation is inherently sensitive to tiny changes of inputs.
        # For example, x = 4.4998 in Pytorch might be calculated as 4.5001 in Tensorflow and rounded to a different integer.
        x = tf.round(x)
        y = tf.round(y)
        return gather(input, y, x, b, h, w, c, padding_mode)


def grid_sample_with_mask(input, grid, canvas=None, mode="bilinear", padding_mode="zeros", align_corners=True):
    output = grid_sample(input, grid=grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    if canvas is None:
        return output
    else:
        input_mask = tf.ones_like(input)
        output_mask = grid_sample(input_mask, grid=grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        output = output * output_mask + canvas * (1 - output_mask)
        return output