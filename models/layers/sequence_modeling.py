import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense


class BidirectionalLSTM(tf.keras.layers.Layer):
    def __init__(self, units, use_dense=True, merge_mode='concat', dropout=0.0, *args, **kwargs):
        super(BidirectionalLSTM, self).__init__(*args, **kwargs)
        assert merge_mode in ('sum', 'mul', 'concat', 'ave', None)
        self.units      = units
        self.use_dense  = use_dense
        self.merge_mode = merge_mode
        self.dropout    = dropout

    def build(self, input_shape):
        self.rnn = Bidirectional(LSTM(units=self.units, return_sequences=True, dropout=self.dropout), 
                                 merge_mode=self.merge_mode,
                                 input_shape=input_shape)
        if self.use_dense:
            self.linear = Dense(units=self.units)

    def call(self, inputs, training=False):
        x = self.rnn(inputs, training=training)
        if hasattr(self, 'linear'):
            x = self.linear(x, training=training)
        return x


class CascadeBidirectionalLSTM(tf.keras.layers.Layer):
    def __init__(self, units, num_layer=2, use_dense=True, merge_mode='concat', dropout=0.0, *args, **kwargs):
        super(CascadeBidirectionalLSTM, self).__init__(*args, **kwargs)
        assert merge_mode in ('sum', 'mul', 'concat', 'ave', None)
        self.units      = units
        self.num_layer  = num_layer
        self.use_dense  = use_dense
        self.merge_mode = merge_mode
        self.dropout    = dropout

    def build(self, input_shape):
        self.block = Sequential([
            BidirectionalLSTM(units=self.units, 
                              use_dense=self.use_dense, 
                              merge_mode=self.merge_mode, 
                              dropout=self.dropout) for i in range(self.num_layer)
        ])
        if self.use_dense:
            self.linear = Dense(units=self.units)

    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        if hasattr(self, 'linear'):
            x = self.linear(x, training=training)
        return x


class MDLSTM(tf.keras.layers.Layer):
    def __init__(self, units, use_dense=True, dropout=0.0, *args, **kwargs):
        super(MDLSTM, self).__init__(*args, **kwargs)
        self.units      = units
        self.use_dense  = use_dense
        self.dropout    = dropout

    def build(self, input_shape):
        self.block = Sequential([
            LSTM(units=self.units*2, return_sequences=True, dropout=self.dropout),
            LSTM(units=self.units*4, return_sequences=True, dropout=self.dropout),
            LSTM(units=self.units*2, return_sequences=True, dropout=self.dropout),
            LSTM(units=self.units, return_sequences=True, dropout=self.dropout),
        ])
        if self.use_dense:
            self.linear = Dense(units=self.units)
            
    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        if hasattr(self, 'linear'):
            x = self.linear(x, training=training)
        return x