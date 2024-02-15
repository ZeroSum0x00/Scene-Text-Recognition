import tensorflow as tf


class EntropyCharAccuracy(tf.keras.metrics.Mean):
    def __init__(self, name="char_accuracy", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, y_true_length, sample_weight=None):
        super().update_state(entropy_character_accuracy(y_true, y_pred, y_true_length), sample_weight)


def entropy_character_accuracy(y_true, y_pred, y_true_length=None):
    def calc(y_true, y_pred, y_true_length):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        if y_pred.dtype != y_true.dtype:
            y_pred = tf.cast(y_pred, dtype=y_true.dtype)

        result = tf.TensorArray(dtype=tf.float32, size=tf.shape(y_true)[0], dynamic_size=True, infer_shape=False)
        def loop_body(i, label, predict, length, result):
            word_length = tf.cast(length[i], dtype=tf.int32)
            matching_label = label[i][:word_length + 1]
            matching_predict = predict[i][:word_length + 1]
            matching_chars = tf.cast(tf.equal(matching_label, matching_predict), dtype=tf.int32)
            matching_chars = tf.reduce_sum(matching_chars) / (word_length + 1)
            matching_chars = tf.cast(matching_chars, dtype=tf.float32)
            result = result.write(i, matching_chars)
            return i+1, label, predict, length, result

        bs = tf.shape(y_true)[0]
        cond = lambda i, *args: i < bs
        _, _, _, _, result = tf.while_loop(cond, loop_body, [0, y_true, y_pred, y_true_length, result])

        matching_labels_count = result.stack()
        return matching_labels_count
            
    if isinstance(y_pred, (tuple, list)):
        accuracy = calc(y_true, y_pred[-1][-1], y_true_length)
    else:
        accuracy = calc(y_true, y_pred, y_true_length)
    return accuracy