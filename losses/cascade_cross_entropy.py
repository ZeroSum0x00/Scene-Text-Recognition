import tensorflow as tf
from tensorflow.keras import losses


class CascadeCrossentropy(tf.keras.losses.Loss):
    def __init__(self, 
                 reduction=tf.losses.Reduction.AUTO, 
                 name='CascadeCrossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.invariant_name = "cascade_cross_entropy"
        self.coefficient = 1
        self.bce = losses.CategoricalCrossentropy()

    def _flatten(self, label, length):
        result = tf.TensorArray(dtype=tf.float32, size=tf.shape(label)[0], dynamic_size=True, infer_shape=False)
        def loop_body(i, lb, le, re):
            flat_value = tf.cast(lb[i][:le[i] + 1], dtype=tf.float32)
            re = re.write(i, flat_value)
            return i+1, lb, le, re

        bs = tf.shape(label)[0]
        cond = lambda i, *args: i < bs
        _, _, _, result = tf.while_loop(cond, loop_body, [0, label, length, result])
        result = result.concat()
        return result

    def cascade_calc(self, y_true, y_pred, label_length):
        if isinstance(y_pred, (tuple, list)):
            y_pred = tf.concat(y_pred, axis=0)

        iter_size = tf.shape(y_pred)[0] // tf.shape(y_true)[0]

        y_true, label_length =  tf.cond(pred=iter_size > 1,
                                        true_fn=lambda: [tf.tile(y_true, (iter_size, 1, 1)), tf.tile(label_length, (iter_size,))],
                                        false_fn=lambda: [y_true, label_length])

        if y_pred.dtype != y_true.dtype:
            y_true = tf.cast(y_true, dtype=y_pred.dtype)
        
        y_true = self._flatten(y_true, label_length)
        y_pred = self._flatten(y_pred, label_length)

        return self.bce(y_true, y_pred) * self.coefficient

    def __call__(self, y_true, y_pred, label_length, sample_weight=None):
        if isinstance(y_pred, (tuple, list)):
            loss = tf.reduce_sum([self.cascade_calc(y_true, pred, label_length) for pred in y_pred])
        else:
            loss = self.cascade_calc(y_true, y_pred, label_length)
        return loss