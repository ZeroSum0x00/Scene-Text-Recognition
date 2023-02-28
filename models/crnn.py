import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from models.layers import BidirectionalLSTM
from utils.train_processing import losses_prepare


class CRNN(tf.keras.Model):
    def __init__(self, backbone, num_filters, hidden_dim, n_classes, *args, **kwargs):
        super(CRNN, self).__init__(*args, **kwargs)
        self.backbone    = backbone
        self.num_filters = num_filters
        self.hidden_dim  = hidden_dim
        self.n_classes   = n_classes
        
    def build(self, input_shape):
        self.map_to_sequence    = Reshape(target_shape=(-1, self.num_filters[-1]))
        self.sequence_modeling  = BidirectionalLSTM(self.hidden_dim)
        self.sequence_modeling2 = BidirectionalLSTM(self.hidden_dim)
        self.predictor          = Dense(units=self.n_classes)
        self.final_activation   = Softmax()
        
    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.map_to_sequence(x)
        x = self.sequence_modeling(x)
        x = self.sequence_modeling2(x)
        x = self.predictor(x)
        x = self.final_activation(x)
        return x

    @tf.function
    def predict(self, inputs):
        y_pred         = self(inputs, training=False)
        preds          = tf.math.argmax(y_pred, axis=2)
        preds_max_prob = tf.reduce_max(y_pred, axis=2)
        batch_length   = tf.cast(tf.shape(y_pred)[0], dtype="int32")
        preds_length   = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        preds_length   = preds_length * tf.ones(shape=(batch_length), dtype="int32")
        return preds, preds_length, preds_max_prob
    
    def calc_loss(self, y_true, y_pred, lenghts, loss_object):
        ctc_loss = losses_prepare(loss_object)
        loss = 0
        if ctc_loss:
            loss += ctc_loss(y_true, y_pred, lenghts)
        return loss
