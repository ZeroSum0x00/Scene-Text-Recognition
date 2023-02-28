import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional


class BidirectionalLSTM(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, *args, **kwargs):
        super(BidirectionalLSTM, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.rnn = Bidirectional(LSTM(units=self.hidden_dim, return_sequences=True), input_shape=input_shape)

    def call(self, inputs, training=False):
        x = self.rnn(inputs)
        return x
