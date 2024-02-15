import tensorflow as tf
from tensorflow.keras import losses


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 reduction=tf.losses.Reduction.AUTO, 
                 name='CTCLoss'):
        super(CTCLoss, self).__init__(reduction=reduction, name=name)
        self.invariant_name = "ctc_loss"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, label_length, sample_weight=None):        
        y_true_length = tf.expand_dims(label_length, axis=1)
        y_pred_shape = tf.shape(y_pred)
        y_pred_length = tf.fill(dims=[y_pred_shape[0], 1], value=y_pred_shape[1])
        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, y_pred_length, y_true_length)
        # loss = tf.nn.ctc_loss(y_true, y_pred, tf.reshape(y_pred_length, -1), tf.reshape(y_true_length, -1), logits_time_major=False)
        return tf.reduce_mean(tf.squeeze(loss) * self.coefficient)
