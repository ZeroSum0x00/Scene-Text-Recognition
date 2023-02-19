import tensorflow as tf
from tensorflow.keras import backend as K


DECODED_PADDING_CONSTANT = -1


class CTCAccuracy(tf.keras.metrics.Metric):
    def __init__(self, 
                 y_true_padding_const=None, 
                 name="CTCAccuracy", **kwargs):
        super(CTCAccuracy, self).__init__(name=name, **kwargs)
        self.y_true_padding_const = y_true_padding_const
        self.accuracy = self.add_weight('accuracy', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        y_true_shape = tf.shape(y_true)

        y_pred_length = tf.fill(dims=[y_pred_shape[0],], value=y_pred_shape[1])

        y_pred = tf.keras.backend.ctc_decode(y_pred, y_pred_length)[0][0]
        y_pred_shape = y_pred_shape[:-1]  # y_pred shape after the decoding loses the last dimension (classes count)

        # If y_true is padded, update it to use the same padding constant as y_pred_decoded
        if self.y_true_padding_const is not None and self.y_true_padding_const != DECODED_PADDING_CONSTANT:
            y_true = tf.where(tf.not_equal(y_true, self.y_true_padding_const), y_true, tf.fill(y_true_shape, DECODED_PADDING_CONSTANT))

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
        correct_predictions = tf.cast(correct_predictions, tf.float32)
        score = tf.reduce_mean(correct_predictions)
        self.accuracy.assign_add(score)
        
    def reset_state(self):
        self.accuracy.assign(0.)

    def result(self):
        return self.accuracy
