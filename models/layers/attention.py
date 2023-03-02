import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTMCell


class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_embeddings, *args, **kwargs):
        super(AttentionCell, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        
    def build(self, input_shape):
        self.i2h = Dense(units=self.hidden_dim, use_bias=False)
        self.h2h = Dense(units=self.hidden_dim)
        self.score = Dense(units=1, use_bias=False)
        self.rnn = LSTMCell(self.hidden_dim)                         # https://stackoverflow.com/questions/54767816/how-exactly-does-lstmcell-from-tensorflow-operates
    
    def call(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), axis=1)

        e = self.score(tf.math.tanh(batch_H_proj + prev_hidden_proj))
        alpha = tf.nn.softmax(e, axis=1)

        alpha_transposed = tf.transpose(alpha, perm=[0, 2, 1])
        context = tf.linalg.matmul(alpha_transposed, batch_H)
        context = tf.squeeze(context, axis=1)
        concat_context = tf.concat([context, char_onehots], axis=1)
        _, cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
      
      
class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_classes, batch_max_length, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_max_length = batch_max_length
        
    def build(self, input_shape):
        self.attention_cell = AttentionCell(self.hidden_dim, self.num_classes)
        self.generator = Dense(units=self.num_classes)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_char = tf.expand_dims(input_char, axis=-1)
        input_char = tf.cast(input_char, dtype=tf.int32)
        bs = tf.shape(input_char)[0]
        one_hot = tf.one_hot(input_char, onehot_dim)
        one_hot = tf.reshape(one_hot, [bs, onehot_dim])
        return one_hot

    def call(self, inputs, text, training=False):
        bs = tf.shape(inputs)[0]
        num_steps = self.batch_max_length + 1

        # output_hiddens = tf.fill(dims=[bs, num_steps, self.hidden_dim], value=0.).numpy()
        output_hiddens = np.full([bs, num_steps, self.hidden_dim], 0.)

        hidden = (tf.fill(dims=[bs, self.hidden_dim], value=0.),
                  tf.fill(dims=[bs, self.hidden_dim], value=0.))
        if training:
            output_list = []

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens[:, i, :] = hidden[0]

            probs = self.generator(output_hiddens)
        else:
            targets = tf.fill(dims=[bs], value=0)
            # probs = tf.fill(dims=[bs, num_steps, self.num_classes], value=0.)
            probs = np.full([bs, num_steps, self.num_classes], 0.)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                next_input = tf.math.argmax(probs_step, axis=1)
                targets = next_input
        return probs  # batch_size x num_steps x num_classes
