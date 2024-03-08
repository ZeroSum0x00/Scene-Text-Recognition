import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import backend as K


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 reduction=tf.losses.Reduction.AUTO, 
                 name='CTCLoss'):
        super(CTCLoss, self).__init__(reduction=reduction, name=name)
        self.invariant_name = "ctc_loss"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, label_length, sample_weight=None):
        if label_length.dtype != tf.int32:
            label_length = tf.cast(label_length, dtype=tf.int32)

        y_pred_shape = tf.shape(y_pred)
        y_pred_length = tf.fill(dims=[y_pred_shape[0]], value=y_pred_shape[1])
        
        sparse_labels = K.ctc_label_dense_to_sparse(y_true, label_length)
        sparse_labels = tf.cast(sparse_labels, tf.int32)

        y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())

        loss = tf.compat.v1.nn.ctc_loss(inputs=y_pred, 
                                        labels=sparse_labels, 
                                        sequence_length=y_pred_length, 
                                        ignore_longer_outputs_than_inputs=True)
        return tf.reduce_mean(loss * self.coefficient)