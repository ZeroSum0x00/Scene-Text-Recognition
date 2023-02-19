"""CTC accuracy"""
import tensorflow as tf


# An unknown character token returned by ctc_decode(), which also serves as a padding constant
DECODED_PADDING_CONSTANT = -1


class CTCAccuracy(tf.keras.metrics.Mean):
    def __init__(self, name="CTCAccuracy", dtype=None):
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(ctc_accuracy(y_true, y_pred), sample_weight)


def ctc_accuracy(y_true, y_pred, y_pred_length=None):
    y_pred_shape = tf.shape(y_pred)
    y_true_shape = tf.shape(y_true)

    y_pred_length = tf.fill((y_pred_shape[0],), y_pred_shape[1])

    y_pred = tf.keras.backend.ctc_decode(y_pred, y_pred_length)[0][0]
    y_pred_shape = y_pred_shape[:-1]  # y_pred shape after the decoding loses the last dimension (classes count)

    # Pad y_true or y_pred so that they have the same max sequence length
    if y_true_shape[1] < y_pred_shape[1]:
        y_true = tf.pad(y_true, paddings=[[0, 0], [0, y_pred_shape[1] - y_true_shape[1]]], constant_values=DECODED_PADDING_CONSTANT)
        y_true_shape = y_pred_shape
    elif y_pred_shape[1] < y_true_shape[1]:
        y_pred = tf.pad(y_pred, paddings=[[0, 0], [0, y_true_shape[1] - y_pred_shape[1]]], constant_values=DECODED_PADDING_CONSTANT)
        y_pred_shape = y_true_shape

    # ctc_decode() returns y_pred of int64 type and if y_true is int32, that will cause an error in tf.equal(), so
    # we need to cast y_pred to the same type
    if y_pred.dtype != y_true.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    matching_labels_count = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int32), axis=1)
    # A prediction considered correct if all its labels match the corresponding true sequence
    correct_predictions = tf.equal(matching_labels_count, y_true_shape[1])

    return tf.cast(correct_predictions, tf.int32)
