import tensorflow as tf
from tensorflow.keras import losses


class AttentionEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 reduction=tf.losses.Reduction.AUTO, 
                 name=None):
        super(AttentionEntropyLoss, self).__init__(reduction=reduction, name=name)
        self.losses = losses.SparseCategoricalCrossentropy(ignore_class=0)
        self.invariant_name = "attention_entropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, label_length, sample_weight=None):
        y_pred = tf.reshape(y_pred, shape=(-1, tf.shape(y_pred)[-1]))
        y_true = tf.reshape(y_true, shape=(-1, 1))
        y_true = tf.squeeze(y_true, axis=-1)
        loss = self.losses(y_true, y_pred)
        return loss * self.coefficient
