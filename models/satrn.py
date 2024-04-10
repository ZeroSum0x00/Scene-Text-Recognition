from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling2D

from models.layers import get_activation_from_name, get_normalizer_from_name, ConvolutionBlock
from utils.train_processing import losses_prepare



def get_sinusoid_encoding_table(embed_dim, n_position):
    """Sinusoid position encoding table."""
    denominator = [1.0 / np.power(10000, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]
    denominator = np.array(denominator).reshape(1, -1)
    pos_tensor = np.arange(n_position).reshape(-1, 1)
    sinusoid_table = pos_tensor * denominator
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table


class Adaptive2DPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 position_size,
                 activation='relu', 
                 normalizer='batch-norm',
                 drop_rate=0.1,
                 *args, **kwargs):
        super(Adaptive2DPositionalEncoding, self).__init__(*args, **kwargs)
        self.embed_dim     = embed_dim
        self.position_size = position_size if isinstance(position_size, (list, tuple)) else (position_size, position_size)
        self.activation    = activation
        self.normalizer    = normalizer
        self.drop_rate     = drop_rate

    def build(self, input_shape):
        h_position_encoder = get_sinusoid_encoding_table(self.embed_dim, self.position_size[0])
        h_position_encoder = np.transpose(h_position_encoder, [1, 0])
        h_position_encoder = h_position_encoder.reshape(1, self.position_size[0], 1, self.embed_dim)
        self.h_position_encoder = tf.convert_to_tensor(h_position_encoder, dtype=tf.float32)
        self.h_position_encoder = tf.Variable(
            initial_value=self.h_position_encoder, trainable=True, name=f'adaptive_2D_positional_encoding/h_position_encoder'
        )

        w_position_encoder = get_sinusoid_encoding_table(self.embed_dim, self.position_size[1])
        w_position_encoder = np.transpose(w_position_encoder, [1, 0])
        w_position_encoder = w_position_encoder.reshape(1, 1, self.position_size[1], self.embed_dim)
        self.w_position_encoder = tf.convert_to_tensor(w_position_encoder, dtype=tf.float32)
        self.w_position_encoder = tf.Variable(
            initial_value=self.w_position_encoder, trainable=True, name=f'adaptive_2D_positional_encoding/w_position_encoder'
        )

        self.h_scale = Sequential([
            Conv2D(filters=self.embed_dim, kernel_size=(1, 1), strides=(1, 1), padding="VALID"),
            get_activation_from_name(self.activation),
            Conv2D(filters=self.embed_dim, kernel_size=(1, 1), strides=(1, 1), padding="VALID"),
            get_activation_from_name('sigmoid')
        ])

        self.w_scale = Sequential([
            Conv2D(filters=self.embed_dim, kernel_size=(1, 1), strides=(1, 1), padding="VALID"),
            get_activation_from_name(self.activation),
            Conv2D(filters=self.embed_dim, kernel_size=(1, 1), strides=(1, 1), padding="VALID"),
            get_activation_from_name('sigmoid')
        ])

        self.pool = GlobalAveragePooling2D(keepdims=True)
        self.dropout = Dropout(self.drop_rate)

    def call(self, inputs, training=False):
        bs = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]

        avg_pool = self.pool(inputs)
        h_pos_encoding =  self.h_scale(avg_pool, training=training) * self.h_position_encoder[:, :h, :, :]
        w_pos_encoding =  self.w_scale(avg_pool, training=training) * self.w_position_encoder[:, :, :w, :]
        x = inputs + h_pos_encoding + w_pos_encoding
        x = self.dropout(x, training=training)
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 key_dim=64,
                 value_dim=64,
                 qkv_bias=False,
                 drop_rate=0.1,
                 *args, **kwargs):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.key_dim     = key_dim
        self.value_dim   = value_dim
        self.qkv_bias    = qkv_bias
        self.drop_rate   = drop_rate
        self.temperature = key_dim**0.5

    def build(self, input_shape):
        self.dim_k = self.num_heads * self.key_dim
        self.dim_v = self.num_heads * self.value_dim

        self.linear_q = Dense(self.dim_k, use_bias=self.qkv_bias)
        self.linear_k = Dense(self.dim_k, use_bias=self.qkv_bias)
        self.linear_v = Dense(self.dim_v, use_bias=self.qkv_bias)

        self.fc = Dense(self.embed_dim, use_bias=self.qkv_bias)
        self.proj_drop = Dropout(self.drop_rate)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        attn = tf.matmul(query / self.temperature, key, transpose_b=True)
        if mask is not None:
            if tf.rank(mask) == 3:
                mask = mask[:, tf.newaxis, :, :]
            elif tf.rank(mask) == 2:
                mask = mask[:, tf.newaxis, tf.newaxis, :]

            if mask.dtype == tf.bool:
                attn = tf.where(mask, attn, -1e30)
            else:
                attn += mask

        weights = tf.nn.softmax(attn, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def call(self, inputs, mask=None, training=False):
        query, key, value = inputs
        len_q = tf.shape(query)[1]
        len_k = tf.shape(value)[1]

        query = self.linear_q(query, training=training)
        query = tf.reshape(query, shape=[-1, self.num_heads, self.key_dim, len_q])
        query = tf.transpose(query, perm=[0, 1, 3, 2])

        key = self.linear_k(key, training=training)
        key = tf.reshape(key, shape=[-1, self.num_heads, self.key_dim, len_k])
        key = tf.transpose(key, perm=[0, 1, 3, 2])

        value = self.linear_q(value, training=training)
        value = tf.reshape(value, shape=[-1, self.num_heads, self.value_dim, len_k])
        value = tf.transpose(value, perm=[0, 1, 3, 2])

        attn_out, _ = self.scaled_dot_product_attention(query, key, value, mask=mask)
        attn_out = tf.transpose(attn_out, perm=[0, 1, 3, 2])
        attn_out = tf.reshape(attn_out, shape=[-1, len_q, self.dim_v])
        attn_out = self.fc(attn_out, training=training)
        attn_out = self.proj_drop(attn_out, training=training)
        return attn_out


class LocalityAwareFeedforward(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 activation='relu', 
                 normalizer='batch-norm',
                 drop_rate=0.1,
                 *args, **kwargs):
        super(LocalityAwareFeedforward, self).__init__(*args, **kwargs)
        self.filters = filters
        self.activation    = activation
        self.normalizer    = normalizer
        self.drop_rate = drop_rate

    def build(self, input_shape):
        out_filters = input_shape[-1]
        self.conv1 = ConvolutionBlock(filters=self.filters,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="VALID",
                                      use_bias=False,
                                      activation=self.activation,
                                      normalizer=self.normalizer)
        self.depthwise_conv = ConvolutionBlock(filters=self.filters,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding="SAME",
                                               use_bias=False,
                                               groups=self.filters,
                                               activation=self.activation,
                                               normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(filters=out_filters,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="VALID",
                                      use_bias=False,
                                      activation=self.activation,
                                      normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.depthwise_conv(x, training=training)
        x = self.conv2(x, training=training)
        return x


class SATRNEncodeLayer(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 forward_dim,
                 input_resolution,
                 num_heads=8,
                 key_dim=64,
                 value_dim=64,
                 qkv_bias=False,
                 activation='relu',
                 local_normalizer='batch-norm',
                 global_normalizer='layer-norm',
                 drop_rate=0.1,
                 *args, **kwargs):
        super(SATRNEncodeLayer, self).__init__(*args, **kwargs)
        self.embed_dim         = embed_dim
        self.forward_dim       = forward_dim
        self.input_resolution  = input_resolution
        self.num_heads         = num_heads
        self.key_dim           = key_dim
        self.value_dim         = value_dim
        self.qkv_bias          = qkv_bias
        self.activation        = activation
        self.local_normalizer  = local_normalizer
        self.global_normalizer = global_normalizer
        self.drop_rate         = drop_rate

    def build(self, input_shape):
        self.norm1 = get_normalizer_from_name(self.global_normalizer)
        self.attn  = MultiHeadAttention(embed_dim=self.embed_dim,
                                        num_heads=self.num_heads,
                                        key_dim=self.key_dim,
                                        value_dim=self.value_dim,
                                        qkv_bias=self.qkv_bias,
                                        drop_rate=self.drop_rate)
        self.norm2 = get_normalizer_from_name(self.global_normalizer)
        self.feed_forward = LocalityAwareFeedforward(self.forward_dim, 
                                                     activation=self.activation,
                                                     normalizer=self.local_normalizer,
                                                     drop_rate=self.drop_rate)

    def call(self, inputs, mask=None, training=False):
        H, W = self.input_resolution
        c = tf.shape(inputs)[-1]
        hw = tf.shape(inputs)[1]
        x = self.norm1(inputs, training=training)
        x = inputs + self.attn([x, x, x], mask=mask)
        residual = x
        x = self.norm2(x, training=training)
        # x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, shape=[-1, H, W, c])
        x = self.feed_forward(x, training=training)
        x = tf.reshape(x, shape=[-1, hw, c])
        x += residual
        return x


class SATRNEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 forward_dim,
                 n_position,
                 iters=12,
                 num_heads=8,
                 key_dim=64,
                 value_dim=64,
                 qkv_bias=False,
                 activation='relu',
                 local_normalizer='batch-norm',
                 global_normalizer='layer-norm',
                 drop_rate=0.1,
                 *args, **kwargs):
        super(SATRNEncoder, self).__init__(*args, **kwargs)
        self.embed_dim         = embed_dim
        self.forward_dim       = forward_dim
        self.n_position        = n_position
        self.iters             = iters
        self.num_heads         = num_heads
        self.key_dim           = key_dim
        self.value_dim         = value_dim
        self.qkv_bias          = qkv_bias
        self.activation        = activation
        self.local_normalizer  = local_normalizer
        self.global_normalizer = global_normalizer
        self.drop_rate         = drop_rate

    def build(self, input_shape):
        self.input_resolution = input_shape[1:-1]
        self.position_enc = Adaptive2DPositionalEncoding(embed_dim=self.embed_dim,
                                                         position_size=self.n_position,
                                                         activation=self.activation, 
                                                         normalizer=self.local_normalizer,
                                                         drop_rate=self.drop_rate)
        self.block_layers = [SATRNEncodeLayer(embed_dim=self.embed_dim,
                                              forward_dim=self.forward_dim,
                                              input_resolution=self.input_resolution,
                                              num_heads=self.num_heads,
                                              key_dim=self.key_dim,
                                              value_dim=self.value_dim,
                                              qkv_bias=self.qkv_bias,
                                              activation=self.activation,
                                              local_normalizer=self.local_normalizer,
                                              global_normalizer=self.global_normalizer,
                                              drop_rate=self.drop_rate) for _ in range(self.iters)]
        self.norm = get_normalizer_from_name(self.global_normalizer)

    def call(self, inputs, lenghts=None, training=False):
        bs = tf.shape(inputs)[0]
        if lenghts is None:
            lenghts = tf.ones((bs))

        h, w = self.input_resolution
        feat = self.position_enc(inputs, training=training)
        mask = tf.zeros((bs, h, w))

        for i, lenght in enumerate(lenghts):
            valid_width = min(w, math.ceil(w * lenght))
            cs = tf.range(0, h, dtype=tf.int32)
            ds = tf.range(0, valid_width, dtype=tf.int32)
            coords_b, coords_c, coords_d = tf.meshgrid(i, cs, ds, indexing='ij')
            relative_coords_table = tf.stack([coords_b, coords_c, coords_d])
            relative_coords_table = tf.transpose(relative_coords_table, [1, 2, 3, 0])
            relative_coords_table = tf.reshape(relative_coords_table, [-1, 3])
            updates = tf.ones((h*valid_width,))
            mask = tf.tensor_scatter_nd_add(mask, relative_coords_table, updates)

        mask = tf.reshape(mask, [bs, h * w])
        mask = tf.cast(mask, dtype=tf.bool)
        feat = tf.reshape(feat, [bs, h*w, -1])

        for layer in self.block_layers:
            feat = layer(feat, mask, training=training)
        feat = self.norm(feat, training=training)
        return feat


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 n_position,
                 drop_rate=0.1,
                 *args, **kwargs):
        super(PositionalEncoding, self).__init__(*args, **kwargs)
        self.embed_dim        = embed_dim
        self.n_position       = n_position
        self.drop_rate        = drop_rate

    def build(self, input_shape):
        position_table = get_sinusoid_encoding_table(self.embed_dim, self.n_position)
        position_table = tf.expand_dims(position_table, axis=0)
        position_table = tf.convert_to_tensor(position_table)
        self.position_table = tf.Variable(
            initial_value=position_table,
            trainable=True,
            name=f'positional_encoding/position_table'
        )
        self.dropout = Dropout(self.drop_rate)

    def call(self, inputs, training=False):
        if self.position_table.dtype != inputs.dtype:
            self.position_table = tf.cast(self.position_table, dtype=inputs.dtype)

        x = inputs + self.position_table[:, :tf.shape(inputs)[1]]
        x = self.dropout(x, training=training)
        return x


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 activation='gelu',
                 drop_rate=0.1,
                 *args, **kwargs):
        super(PositionwiseFeedForward, self).__init__(*args, **kwargs)
        self.embed_dim  = embed_dim
        self.activation = activation
        self.drop_rate  = drop_rate

    def build(self, input_shape):
        self.w_1     = Dense(self.embed_dim)
        self.w_2     = Dense(input_shape[-1])
        self.activ   = get_activation_from_name(self.activation)
        self.dropout = Dropout(self.drop_rate)

    def call(self, inputs, training=False):
        x = self.w_1(inputs, training=training)
        x = self.activ(x, training=training)
        x = self.w_2(x, training=training)
        x = self.dropout(x, training=training)
        return x


class SATRNDecodeLayer(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 forward_dim,
                 num_heads=8,
                 key_dim=64,
                 value_dim=64,
                 qkv_bias=False,
                 use_prenorm=False,
                 activation='relu',
                 normalizer='layer-norm',
                 drop_rate=0.1,
                 *args, **kwargs):
        super(SATRNDecodeLayer, self).__init__(*args, **kwargs)
        self.embed_dim   = embed_dim
        self.forward_dim = forward_dim
        self.num_heads   = num_heads
        self.key_dim     = key_dim
        self.value_dim   = value_dim
        self.qkv_bias    = qkv_bias
        self.use_prenorm = use_prenorm
        self.activation  = activation
        self.normalizer  = normalizer
        self.drop_rate   = drop_rate

    def build(self, input_shape):
        self.self_attn = MultiHeadAttention(embed_dim=self.embed_dim,
                                            num_heads=self.num_heads,
                                            key_dim=self.key_dim,
                                            value_dim=self.value_dim,
                                            qkv_bias=self.qkv_bias,
                                            drop_rate=self.drop_rate)
        self.enc_attn = MultiHeadAttention(embed_dim=self.embed_dim,
                                            num_heads=self.num_heads,
                                            key_dim=self.key_dim,
                                            value_dim=self.value_dim,
                                            qkv_bias=self.qkv_bias,
                                            drop_rate=self.drop_rate)
        self.mlp = PositionwiseFeedForward(embed_dim=self.embed_dim,
                                           activation=self.activation,
                                           drop_rate=self.drop_rate)
        self.norm1 = get_normalizer_from_name(self.normalizer)
        self.norm2 = get_normalizer_from_name(self.normalizer)
        self.norm3 = get_normalizer_from_name(self.normalizer)

    def call(self, inputs, dec_attn_mask=None, enc_attn_mask=None, training=False):
        dec_input, enc_output = inputs
        if self.use_prenorm:
            dec_input_norm = self.norm1(dec_input, training=training)
            dec_attn_out = self.self_attn([dec_input_norm, dec_input_norm, dec_input_norm], mask=dec_attn_mask, training=training)
            dec_attn_out += dec_input
            enc_dec_attn_in = self.norm2(dec_attn_out, training=training)
            enc_dec_attn_out = self.enc_attn([enc_dec_attn_in, enc_output, enc_output], mask=enc_attn_mask, training=training)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm3(enc_dec_attn_out, training=training)
            mlp_out = self.mlp(enc_dec_attn_out, training=training)
            mlp_out += enc_dec_attn_out
        else:
            dec_attn_out = self.self_attn([dec_input, dec_input, dec_input], mask=dec_attn_mask, training=training)
            dec_attn_out += dec_input
            dec_attn_out = self.norm1(dec_attn_out, training=training)
            enc_dec_attn_out = self.enc_attn([dec_attn_out, enc_output, enc_output], mask=enc_attn_mask, training=training)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm2(enc_dec_attn_out, training=training)
            mlp_out = self.mlp(enc_dec_attn_out, training=training)
            mlp_out += enc_dec_attn_out
            mlp_out = self.norm3(mlp_out, training=training)
        return mlp_out


class SATRNDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 forward_dim,
                 n_position=200,
                 iters=6,
                 num_heads=8,
                 key_dim=64,
                 value_dim=64,
                 qkv_bias=False,
                 batch_max_length=25,
                 n_classes=1000,
                 start_idx=1,
                 blank_id=92,
                 activation='relu',
                 normalizer='layer-norm',
                 drop_rate=0.1,
                 *args, **kwargs):
        super(SATRNDecoder, self).__init__(*args, **kwargs)
        self.embed_dim        = embed_dim
        self.forward_dim      = forward_dim
        self.n_position       = n_position
        self.iters            = iters
        self.num_heads        = num_heads
        self.key_dim          = key_dim
        self.value_dim        = value_dim
        self.qkv_bias         = qkv_bias
        self.batch_max_length = batch_max_length
        self.n_classes        = n_classes
        self.start_idx        = start_idx
        self.blank_id         = blank_id
        self.activation       = activation
        self.normalizer       = normalizer
        self.drop_rate        = drop_rate

    def build(self, input_shape):
        self.trg_word_emb = Embedding(self.n_classes, self.embed_dim)
        self.position_enc = PositionalEncoding(self.embed_dim, self.n_position, drop_rate=self.drop_rate)
        self.dropout = Dropout(self.drop_rate)
        self.block_layers = [SATRNDecodeLayer(embed_dim=self.embed_dim,
                                              forward_dim=self.forward_dim,
                                              num_heads=self.num_heads,
                                              key_dim=self.key_dim,
                                              value_dim=self.value_dim,
                                              qkv_bias=self.qkv_bias,
                                              use_prenorm=False,
                                              activation=self.activation,
                                              normalizer=self.normalizer,
                                              drop_rate=self.drop_rate) for _ in range(self.iters)]
        self.norm = get_normalizer_from_name(self.normalizer)
        self.classifier = Dense(self.n_classes - 1)

    def get_mask(self, logit, lenghtss=None):
        N, T, _ = logit.shape
        mask = None
        if lenghtss is not None:
            mask = tf.zeros((N, T))
            for i, lenghts in enumerate(lenghtss):
                valid_width = min(T, math.ceil(T * lenghts))
                cs = tf.range(0, valid_width, dtype=tf.int32)
                coords_b, coords_c = tf.meshgrid(i, cs, indexing='ij')
                relative_coords_table = tf.stack([coords_b, coords_c])
                relative_coords_table = tf.transpose(relative_coords_table, [1, 2, 0])
                relative_coords_table = tf.squeeze(relative_coords_table, axis=0)
                relative_coords_table = tf.reshape(relative_coords_table, [-1, 2])
                updates = tf.ones((valid_width,))
                mask = tf.tensor_scatter_nd_add(mask, relative_coords_table, updates)
        return mask

    def get_subsequent_mask(self, seq):
        """For masking out the subsequent info."""
        len_s = tf.shape(seq)[1]
        subsequent_mask = 1 - tf.experimental.numpy.triu(tf.ones((len_s, len_s)), k=1)
        subsequent_mask = tf.cast(subsequent_mask, dtype=tf.bool)
        return subsequent_mask

    def calc_attention(self, trg_seq, src, src_mask=None, training=False):
        trg_embedding = self.trg_word_emb(trg_seq, training=training)
        trg_pos_encoded = self.position_enc(trg_embedding, training=training)
        tgt = self.dropout(trg_pos_encoded, training=training)

        trg_mask1 = tf.not_equal(trg_seq, self.blank_id)
        trg_mask1 = tf.expand_dims(trg_mask1, axis=1)
        trg_mask2 = self.get_subsequent_mask(trg_seq)
        trg_mask = trg_mask1 & trg_mask2

        for layer in self.block_layers:
            tgt = layer([tgt, src], dec_attn_mask=trg_mask, enc_attn_mask=src_mask, training=training)

        tgt = self.norm(tgt, training=training)
        return tgt

    def call(self, inputs, labels, lenghts, training=False):
        src_mask = self.get_mask(inputs, lenghts)
        bs = tf.shape(inputs)[0]
        if training:
            attn_output = self.calc_attention(labels, inputs, src_mask=src_mask, training=training)
            outputs = self.classifier(attn_output, training=training)
        else:
            init_target_seq = tf.fill(dims=[bs, self.batch_max_length + 1], value=self.blank_id)
            init_target_seq = tf.cast(init_target_seq, dtype=tf.int64)
            br = tf.range(0, bs, dtype=tf.int32)
            coords_b, coords_c = tf.meshgrid(br, 0, indexing='ij')
            relative_coords_table = tf.stack([coords_b, coords_c])
            relative_coords_table = tf.transpose(relative_coords_table, [1, 2, 0])
            relative_coords_table = tf.squeeze(relative_coords_table, axis=1)
            relative_coords_table = tf.reshape(relative_coords_table, [-1, 2])
            updates = tf.fill(dims=[bs,], value=self.start_idx)
            updates = tf.cast(updates, dtype=tf.int64)
            init_target_seq = tf.tensor_scatter_nd_update(init_target_seq, relative_coords_table, updates)

            outputs = []
            for step in range(0, self.batch_max_length):
                decoder_output = self.calc_attention(init_target_seq, inputs, src_mask=src_mask, training=training)
                step_result = self.classifier(decoder_output[:, step, :], training=training)
                step_result = tf.nn.softmax(step_result, axis=-1)
                outputs.append(step_result)
                step_max_index = tf.argmax(step_result, axis=-1)
                cs = tf.range(step + 1, step + 2, dtype=tf.int32)
                coords_b, coords_c = tf.meshgrid(br, cs, indexing='ij')
                relative_coords_table = tf.stack([coords_b, coords_c])
                relative_coords_table = tf.transpose(relative_coords_table, [1, 2, 0])
                relative_coords_table = tf.squeeze(relative_coords_table, axis=1)
                relative_coords_table = tf.reshape(relative_coords_table, [-1, 2])
                init_target_seq = tf.tensor_scatter_nd_update(init_target_seq, relative_coords_table, step_max_index)

        outputs = tf.stack(outputs, axis=1)
        return outputs


class SATRN(tf.keras.Model):
    def __init__(self,
                 backbone,
                 transform_net=None,
                 embed_dim=512,
                 forward_dim=1024,
                 encode_n_position=100,
                 decode_n_position=200,
                 iters=6,
                 num_heads=8,
                 key_dim=64,
                 value_dim=64,
                 qkv_bias=False,
                 batch_max_length=25,
                 n_classes=1000,
                 start_idx=1,
                 blank_id=92,
                 drop_rate=0.1,
                 *args, **kwargs):
        super(SATRN, self).__init__(*args, **kwargs)
        self.backbone          = backbone
        self.transform_net     = transform_net
        self.embed_dim         = embed_dim
        self.forward_dim       = forward_dim
        self.encode_n_position = encode_n_position
        self.decode_n_position = decode_n_position
        self.iters             = iters
        self.num_heads         = num_heads
        self.key_dim           = key_dim
        self.value_dim         = value_dim
        self.qkv_bias          = qkv_bias
        self.batch_max_length  = batch_max_length
        self.n_classes         = n_classes
        self.start_idx         = start_idx
        self.blank_id          = blank_id
        self.drop_rate         = drop_rate

    def build(self, input_shape):
        self.encoder = SATRNEncoder(embed_dim=self.embed_dim,
                                    forward_dim=self.forward_dim,
                                    n_position=self.encode_n_position,
                                    iters=self.iters,
                                    num_heads=self.num_heads,
                                    key_dim=self.key_dim,
                                    value_dim=self.value_dim,
                                    qkv_bias=self.qkv_bias,
                                    activation='relu',
                                    local_normalizer='batch-norm',
                                    global_normalizer='layer-norm',
                                    drop_rate=self.drop_rate)
        self.decoder = SATRNDecoder(embed_dim=self.embed_dim,
                                    forward_dim=self.forward_dim,
                                    n_position=self.decode_n_position,
                                    iters=self.iters,
                                    num_heads=self.num_heads,
                                    key_dim=self.key_dim,
                                    value_dim=self.value_dim,
                                    qkv_bias=self.qkv_bias,
                                    batch_max_length=self.batch_max_length,
                                    n_classes=self.n_classes,
                                    start_idx=self.start_idx,
                                    blank_id=self.blank_id,
                                    activation='relu',
                                    normalizer='layer-norm',
                                    drop_rate=self.drop_rate)

    def call(self, inputs, labels=None, lenghts=None, training=False):
        if self.transform_net is not None:
            inputs = self.transform_net(inputs, training=training)

        x = self.backbone(inputs, training=training)
        x = self.encoder(x, lenghts, training=training)
        x = self.decoder(x, labels, lenghts, training=training)
        return x

    def calc_loss(self, y_true, y_pred, lenghts, loss_object):
        losses = losses_prepare(loss_object)
        loss_value = 0
        if losses:
            for loss in losses:
                loss_value += loss(y_true, y_pred, lenghts)
        return loss_value