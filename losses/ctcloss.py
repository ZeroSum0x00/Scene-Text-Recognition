import tensorflow as tf
from tensorflow.keras import losses


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 y_true_padding_const=None, 
                 reduction=tf.losses.Reduction.AUTO, 
                 name=None):
        super(CTCLoss, self).__init__(reduction=reduction, name=name)
        self.y_true_padding_const = y_true_padding_const
        self.invariant_name = "CTCLoss"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, label_length):
        y_true_length = tf.expand_dims(label_length, axis=1)
        y_pred_shape = tf.shape(y_pred)
        y_pred_length = tf.fill(dims=[y_pred_shape[0], 1], value=y_pred_shape[1])
        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, y_pred_length, y_true_length)
        return tf.squeeze(loss) * self.coefficient
