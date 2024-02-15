import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTMCell


class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, *args, **kwargs):
        super(AttentionCell, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        self.i2h = Dense(units=self.hidden_dim, use_bias=False)
        self.h2h = Dense(units=self.hidden_dim)
        self.score = Dense(units=1, use_bias=False)
        self.rnn = LSTMCell(self.hidden_dim)                         # https://stackoverflow.com/questions/54767816/how-exactly-does-lstmcell-from-tensorflow-operates
    
    def call(self, inputs, training=False):
        prev_hidden, batch_H, char_onehots = inputs
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), axis=1)

        e = self.score(tf.nn.tanh(batch_H_proj + prev_hidden_proj))
        alpha = tf.nn.softmax(e, axis=1)

        alpha_transposed = tf.transpose(alpha, perm=[0, 2, 1])
        context = tf.matmul(alpha_transposed, batch_H)
        context = tf.squeeze(context, axis=1)
        concat_context = tf.concat([context, char_onehots], axis=1)
        _, cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
      
      
class STRAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_classes, batch_max_length=25, *args, **kwargs):
        super(STRAttention, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_max_length = batch_max_length
        
    def build(self, input_shape):
        self.attention_cell = AttentionCell(self.hidden_dim)
        self.generator = Dense(units=self.num_classes)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_char = tf.expand_dims(input_char, axis=-1)
        input_char = tf.cast(input_char, dtype=tf.int32)
        bs = tf.shape(input_char)[0]
        one_hot = tf.one_hot(input_char, onehot_dim)
        one_hot = tf.reshape(one_hot, [bs, onehot_dim])
        return one_hot

    def call(self, inputs, training=False):
        x, text = inputs
        bs = tf.shape(x)[0]
        num_steps = self.batch_max_length + 1

        output_hiddens = tf.fill(dims=[bs, num_steps, self.hidden_dim], value=0.)
        
        hidden = (tf.fill(dims=[bs, self.hidden_dim], value=0.),
                  tf.fill(dims=[bs, self.hidden_dim], value=0.))
        
        if training:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell([hidden, x, char_onehots], training=training)
                hidden_result = hidden[0]
                hidden_shape  = tf.shape(hidden_result)
                hidden_result = tf.reshape(hidden_result, (-1, 1))
                hidden_result = tf.squeeze(hidden_result, axis=-1)
                bs = tf.range(0, hidden_shape[0], dtype=tf.int32)
                cs = tf.range(0, hidden_shape[1], dtype=tf.int32)
                coords_b, coords_c, coords_d = tf.meshgrid(bs, i, cs, indexing='ij')
                relative_coords_table = tf.stack([coords_b, coords_c, coords_d])
                relative_coords_table = tf.transpose(relative_coords_table, [1, 2, 3, 0])
                relative_coords_table = tf.squeeze(relative_coords_table, axis=1)
                relative_coords_table = tf.reshape(relative_coords_table, [-1, 3])
                output_hiddens = tf.tensor_scatter_nd_add(output_hiddens, relative_coords_table, hidden_result)

            probs = self.generator(output_hiddens)
        else:
            targets = tf.fill(dims=[bs], value=0)
            probs = tf.fill(dims=[bs, num_steps, self.num_classes], value=0.)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell([hidden, x, char_onehots], training=training)
                probs_step = self.generator(hidden[0])
                prob_shape = tf.shape(probs_step)
                probs_clone = tf.reshape(probs_step, (-1, 1))
                probs_clone = tf.squeeze(probs_clone, axis=-1)
                bs = tf.range(0, prob_shape[0], dtype=tf.int32)
                cs = tf.range(0, prob_shape[1], dtype=tf.int32)
                coords_b, coords_c, coords_d = tf.meshgrid(bs, i, cs, indexing='ij')
                relative_coords_table = tf.stack([coords_b, coords_c, coords_d])
                relative_coords_table = tf.transpose(relative_coords_table, [1, 2, 3, 0])
                relative_coords_table = tf.squeeze(relative_coords_table, axis=1)
                relative_coords_table = tf.reshape(relative_coords_table, [-1, 3])
                probs = tf.tensor_scatter_nd_add(probs, relative_coords_table, probs_clone)

                next_input = tf.argmax(probs_step, axis=1)
                targets = next_input
        return probs  # batch_size x num_steps x num_classes