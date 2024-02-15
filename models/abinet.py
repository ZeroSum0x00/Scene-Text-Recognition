import copy
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import concatenate
from utils.train_processing import losses_prepare
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2


class GetTokenLength(tf.keras.layers.Layer):

    def __init__(self, blank_index=0, *args, **kwargs):
        super(GetTokenLength, self).__init__(*args, **kwargs)
        self.blank_index = blank_index

    def call(self, inputs, training=False):
        out = tf.argmax(inputs, axis=-1)
        out = tf.equal(out, self.blank_index)
        out = tf.cast(out, dtype=tf.int32)
        mask_max_indices = tf.argmax(out, axis=-1)
        mask_max_values = tf.reduce_max(out, axis=-1)
        mask_max_indices = tf.where(tf.equal(mask_max_values, 0), -1, tf.cast(mask_max_indices, dtype=tf.int32))
        return mask_max_indices + 1


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, max_len=5000, drop_rate=0.1, *args, **kwargs):
        super(PositionalEncoding, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.dropout = Dropout(self.drop_rate)

        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        term = np.arange(0, self.embed_dim, 2, dtype=np.float32)
        div_term = np.exp(term * (-math.log(10000.0) / self.embed_dim))

        pe = np.zeros((self.max_len, self.embed_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, axis=0)
        pe = np.transpose(pe, [1, 0 ,2])
        self.pe = tf.Variable(
            initial_value=pe, trainable=True, name=f'positional_encoding/pe'
        )

    def call(self, inputs, training=False):
        bs = tf.shape(inputs)[0]
        x = inputs + tf.cast(self.pe[:bs, :], dtype=inputs.dtype)
        x = self.dropout(x, training=training)
        return x


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, q_bias=True, kv_bias=False, zeros_adding=False, return_weight=True, drop_rate=0.1, *args, **kwargs):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.q_bias        = q_bias
        self.kv_bias       = kv_bias
        self.zeros_adding  = zeros_adding
        self.return_weight = return_weight
        self.drop_rate     = drop_rate
        self.head_dim      = embed_dim // num_heads
        self.scale         = self.head_dim ** -0.5

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 2:
                self.query_project  = Dense(self.embed_dim, use_bias=self.q_bias, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
                self.keyvalue_project = Dense(self.embed_dim * 2, use_bias=self.kv_bias, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
            else:
                self.query_project  = Dense(self.embed_dim, use_bias=self.q_bias, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
                self.key_project    = Dense(self.embed_dim, use_bias=self.kv_bias, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
                self.value_project  = Dense(self.embed_dim, use_bias=self.kv_bias, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        else:
            self.qkv_projection = Dense(self.embed_dim * 3, use_bias=self.q_bias, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.output_dense       = Dense(self.embed_dim, use_bias=True, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))

    def separate_heads(self, x, middle_dim):
        x = tf.reshape(x, (-1, middle_dim, self.head_dim))
        x = tf.transpose(x, perm=[1, 0, 2])
        return x

    def call(self, inputs, attn_mask=None, key_padding_mask=None, training=False):
        if hasattr(self, 'key_project') and hasattr(self, 'value_project'):
            query, key, value = inputs
            bb = tf.shape(query)[1]
            cc = tf.shape(query)[-1]
            query = self.query_project(query, training=training)
            key   = self.key_project(key, training=training)
            value = self.value_project(value, training=training)
        elif hasattr(self, 'keyvalue_project'):
            query, key_value = inputs
            bb = tf.shape(query)[1]
            cc = tf.shape(query)[-1]
            query = self.query_project(query, training=training)
            kv    = self.keyvalue_project(key_value, training=training)
            key, value = tf.split(kv, 2, axis=-1)
        else:
            bb = tf.shape(inputs)[1]
            cc = tf.shape(inputs)[-1]
            qkv = self.qkv_projection(inputs, training=training)
            query, key, value = tf.split(qkv, 3, axis=-1)

        query = query * self.scale
        query = self.separate_heads(query, bb * self.num_heads)
        key   = self.separate_heads(key, bb * self.num_heads)
        value = self.separate_heads(value, bb * self.num_heads)
        key_shape = tf.shape(key)
        dd = key_shape[1]

        if attn_mask is not None:
            if tf.rank(attn_mask) == 2:
                attn_mask = tf.expand_dims(attn_mask, axis=0)
            #     if list(attn_mask.shape) != [1, attn.shape[1], attn.shape[2]]:
            #         raise RuntimeError('The size of the 2D attn_mask is not correct.')
            # elif tf.rank(attn_mask) == 3:
            #     if list(attn_mask.shape) != [bb * self.num_heads, attn.shape[1], attn.shape[2]]:
            #         raise RuntimeError('The size of the 2D attn_mask is not correct.')

        if self.zeros_adding:
            dd += 1
            key = concatenate([key, tf.zeros((key_shape[0], 1) + key_shape[2:], dtype=tf.float32)], axis=1)
            value = concatenate([value, tf.zeros((key_shape[0], 1) + key_shape[2:], dtype=tf.float32)], axis=1)
            if attn_mask is not None:
                padding = [[0, 0],
                           [0, 0],
                           [0, 1]]
                attn_mask = tf.pad(attn_mask, padding)

            if key_padding_mask is not None:
                padding = [[0, 0],
                           [0, 1]]
                key_padding_mask = tf.pad(key_padding_mask, padding)

        attn = tf.matmul(query, key, transpose_b=True)

        if attn_mask is not None:
            if attn_mask.dtype == tf.bool:
                attn = tf.where(attn_mask, -1e30, attn)
            else:
                attn += attn_mask

        if key_padding_mask is not None:
            attn = tf.reshape(attn, [bb, self.num_heads, -1, dd])
            key_padding_mask = key_padding_mask[:, tf.newaxis, tf.newaxis, :]
            if key_padding_mask.dtype == tf.bool:
                attn = tf.where(key_padding_mask, -1e30, attn)
            else:
                attn += key_padding_mask
            attn = tf.reshape(attn, [bb * self.num_heads, -1, dd])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.nn.dropout(attn, rate=self.drop_rate)
        x = tf.matmul(attn, value)
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, bb, cc])
        x = self.output_dense(x, training=training)

        if self.return_weight:
            attn = tf.reshape(attn, [bb, self.num_heads, -1, dd])
            attn = tf.reduce_sum(attn, axis=1)
            return x, attn / self.num_heads
        else:
          return x, None

            
class PositionAttention(tf.keras.layers.Layer):

    def __init__(self, max_length, num_channels=64, return_weight=True, *args, **kwargs):
        super(PositionAttention, self).__init__(*args, **kwargs)
        self.max_length      = max_length
        self.num_channels     = num_channels
        self.return_weight = return_weight

    def _encoder_block(self, filters, kernel_size=(3, 3), strides=(2, 2), padding="SAME"):
        return Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4)),
            BatchNormalization(),
            Activation('relu')
        ])

    def _decoder_block(self, upsamp_size, filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME'):
        return Sequential([
            UpSampling2D(size=upsamp_size, interpolation='nearest'),
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4)),
            BatchNormalization(),
            Activation('relu')
        ])

    def build(self, input_shape):
        c = input_shape[-1]

        self.k_encoder = [
            self._encoder_block(self.num_channels, (3, 3), (1, 2), 'same'),
            self._encoder_block(self.num_channels, (3, 3), (2, 2), 'same'),
            self._encoder_block(self.num_channels, (3, 3), (2, 2), 'same'),
            self._encoder_block(self.num_channels, (3, 3), (2, 2), 'same')
        ]

        self.k_decoder = [
            self._decoder_block(2, self.num_channels, 3, 1, 'same'),
            self._decoder_block(2, self.num_channels, 3, 1, 'same'),
            self._decoder_block(2, self.num_channels, 3, 1, 'same'),
            self._decoder_block((1, 2), c, 3, 1, 'same'),
        ]

        self.pos_encoder = PositionalEncoding(c, max_len=self.max_length, drop_rate=0)
        self.projection = Dense(c, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))

    def call(self, inputs, training=False):
        key = value = inputs
        N = tf.shape(inputs)[0]
        E = tf.shape(inputs)[1]
        H = tf.shape(inputs)[2]
        W = tf.shape(inputs)[3]

        features = []
        for encoder in self.k_encoder:
            key = encoder(key, training=training)
            features.append(key)

        for i, decoder in enumerate(self.k_decoder):
            key = decoder(key, training=training)
            if i <= len(self.k_decoder) - 2:
                key += features[len(self.k_decoder) - 2 - i]

        zeros = tf.zeros((self.max_length, N, W), dtype=tf.float32)
        query = self.pos_encoder(zeros)
        query = tf.transpose(query, [1, 0, 2])
        query = self.projection(query, training=training)

        value = tf.reshape(value, [-1, E * H, W])

        attn_scores = tf.matmul(query, tf.reshape(key, [-1, E * H, W]), transpose_b=True)
        attn_scores = attn_scores / (tf.cast(E, dtype=tf.float32) ** 0.5)
        attn_scores = tf.nn.softmax(attn_scores, axis=-1)

        attn_result = tf.matmul(attn_scores, value)
        if self.return_weight:
            return attn_result, tf.reshape(attn_scores, [-1, E, H, attn_scores.shape[1]])
        else:
            return attn_result, None


class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, out_dim=2048, drop_rate=0.1, *args, **kwargs):
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.out_dim       = out_dim
        self.drop_rate     = drop_rate

    def build(self, input_shape):
        self.attention = MultiHeadAttention(embed_dim=self.embed_dim,
                                            num_heads=self.num_heads,
                                            zeros_adding=False,
                                            return_weight=False,
                                            drop_rate=self.drop_rate)

        self.dropout1 = Dropout(self.drop_rate)
        self.dropout2 = Dropout(self.drop_rate)
        self.dropout3 = Dropout(self.drop_rate)
        self.dense1 = Dense(self.out_dim, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.dense2 = Dense(self.embed_dim, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.activ = Activation('relu')

    def call(self, inputs, attn_mask=None, key_padding_mask=None, training=False):
        x, _ = self.attention(inputs, attn_mask, key_padding_mask, training=training)
        x = self.dropout1(x, training=training)
        x = x + inputs
        x = self.layernorm1(x, training=training)

        y = self.dense1(x, training=training)
        y = self.activ(y, training=training)
        y = self.dropout2(y, training=training)
        y = self.dense2(y, training=training)
        y = self.dropout3(y, training=training)

        x = x + y
        x = self.layernorm2(x, training=training)
        return x


class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, out_dim=2048, activation="relu", auxiliary_attn=True, siamese=False, drop_rate=0.1, *args, **kwargs):
        super(TransformerDecoderLayer, self).__init__(*args, **kwargs)
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.out_dim       = out_dim
        self.activation    = activation
        self.auxiliary_attn     = auxiliary_attn
        self.siamese       = siamese
        self.drop_rate     = drop_rate

    def build(self, input_shape):
        if self.auxiliary_attn:
            self.attention0 = MultiHeadAttention(embed_dim=self.embed_dim,
                                                 num_heads=self.num_heads,
                                                 zeros_adding=False,
                                                 return_weight=False,
                                                 drop_rate=self.drop_rate)
            self.norm0 = LayerNormalization()
            self.dropout0 = Dropout()

        self.attention1 = MultiHeadAttention(embed_dim=self.embed_dim,
                                             num_heads=self.num_heads,
                                             zeros_adding=False,
                                             return_weight=False,
                                             drop_rate=self.drop_rate)

        self.dropout1 = Dropout(self.drop_rate)
        self.norm1 = LayerNormalization()

        self.dense2 = Dense(self.out_dim, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.activ2 = Activation(self.activation)
        self.dropout2 = Dropout(self.drop_rate)

        self.dense3 = Dense(self.embed_dim, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.dropout3 = Dropout(self.drop_rate)
        self.norm3 = LayerNormalization()

        if self.siamese:
            self.attention2 = MultiHeadAttention(embed_dim=self.embed_dim,
                                                 num_heads=self.num_heads,
                                                 zeros_adding=False,
                                                 return_weight=False,
                                                 drop_rate=self.drop_rate)
            self.dropout2 = Dropout(self.drop_rate)

    def call(self, inputs, attn_mask=None, memory_mask=None,
             attn_key_padding_mask=None, memory_key_padding_mask=None,
             siamese=None, siamese_mask=None, siamese_key_padding_mask=None,
             training=False):
        attn, memory = inputs

        if hasattr(self, 'attention0'):
            attn, _ = self.attention0(attn,
                                      attn_mask=attn_mask,
                                      key_padding_mask=attn_key_padding_mask,
                                      training=training)
            attn += self.dropout0(attn, training=training)
            attn = self.norm0(attn, training=training)

        attn1, _ = self.attention1([attn, memory],
                                    attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask,
                                    training=training)

        if hasattr(self, 'attention2'):
            attn, _ = self.attention2([attn, siamese],
                                       attn_mask=siamese_mask,
                                       key_padding_mask=siamese_key_padding_mask,
                                       training=training)
            attn += self.dropout2(attn, training=training)

        attn += self.dropout1(attn1, training=training)
        attn = self.norm1(attn, training=training)

        attn2 = self.dense2(attn, training=training)
        attn2 = self.activ2(attn2, training=training)
        attn2 = self.dropout2(attn2, training=training)
        attn2 = self.dense3(attn2, training=training)

        attn += self.dropout3(attn2, training=training)
        attn = self.norm3(attn, training=training)
        return attn

                 
class BaseVision(tf.keras.layers.Layer):

    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 out_dim, 
                 num_layers=3,
                 max_length=25, 
                 num_classes=37, 
                 blank_index=0, 
                 drop_rate=0.1, 
                 *args, **kwargs):
        super(BaseVision, self).__init__(*args, **kwargs)
        self.embed_dim       = embed_dim
        self.num_heads       = num_heads
        self.out_dim         = out_dim
        self.num_layers      = num_layers
        self.max_length      = max_length + 1
        self.num_classes     = num_classes
        self.blank_index     = blank_index
        self.drop_rate       = drop_rate

    def build(self, input_shape):
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len=8*32, drop_rate=self.drop_rate)
        self.transformer_encoder = Sequential([
            TransformerEncoderLayer(embed_dim=self.embed_dim,
                                    num_heads=self.num_heads,
                                    out_dim=self.out_dim,
                                    drop_rate=self.drop_rate) for i in range(self.num_layers)
        ])
        self.attention  = PositionAttention(self.max_length, return_weight=False)
        self.get_length = GetTokenLength(self.blank_index)
        self.cls        = Dense(self.num_classes, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))

    def call(self, inputs, training=False):
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        c = tf.shape(inputs)[-1]

        x = tf.reshape(inputs, [-1, h*w, c])
        x = tf.transpose(x, [1, 0, 2])
        x = self.pos_encoder(x, training=training)
        x = self.transformer_encoder(x, training=training)
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, h, w, c])
        attn_vecs, _ = self.attention(x, training=training)
        x = self.cls(attn_vecs, training=training)
        lengths = self.get_length(x)
        return x, attn_vecs, lengths


class BCNLanguage(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, out_dim,
                 num_classes=37, max_length=25, num_layers=4,
                 auxiliary_attn=False, activation='relu', drop_rate=0.1, *args, **kwargs):
        super(BCNLanguage, self).__init__(*args, **kwargs)
        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.out_dim        = out_dim
        self.num_classes    = num_classes
        self.max_length     = max_length + 1
        self.num_layers     = num_layers
        self.auxiliary_attn = auxiliary_attn
        self.activation     = activation
        self.drop_rate      = drop_rate

    def build(self, input_shape):
        self.projection    = Dense(self.embed_dim, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.token_encoder = PositionalEncoding(self.embed_dim, max_len=self.max_length, drop_rate=self.drop_rate)
        self.pos_encoder   = PositionalEncoding(self.embed_dim, max_len=self.max_length, drop_rate=0)
        self.transformer_decoder = [
            TransformerDecoderLayer(embed_dim=self.embed_dim,
                                    num_heads=self.num_heads,
                                    out_dim=self.out_dim,
                                    activation=self.activation,
                                    auxiliary_attn=self.auxiliary_attn,
                                    siamese=False,
                                    drop_rate=self.drop_rate) for i in range(self.num_layers)
        ]
        self.final_dense = Dense(self.num_classes, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))

    def _get_padding_mask(self, length, max_length):
        length = tf.expand_dims(length, axis=-1)
        grid = tf.range(0, max_length, dtype=tf.float32)
        grid = tf.expand_dims(grid, axis=0)
        return grid >= tf.cast(length, dtype=grid.dtype)

    def _get_location_mask(self, w):
        mask = tf.eye(w, dtype=tf.float32)
        mask = tf.where(tf.equal(mask, 1), -1e30, 0)
        return mask

    def call(self, inputs, training=False):
        tokens, lengths = inputs
        embed = self.projection(tokens, training=training)
        embed = tf.transpose(embed, [1, 0, 2])
        embed = self.token_encoder(embed, training=training)

        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = tf.zeros_like(embed)
        query = self.pos_encoder(zeros, training=training)

        location_mask = self._get_location_mask(self.max_length)

        for i, decoder in enumerate(self.transformer_decoder):
            query = decoder([query, embed],
                             attn_key_padding_mask=padding_mask,
                             memory_mask=location_mask,
                             memory_key_padding_mask=padding_mask)
        attn = tf.transpose(query, [1, 0, 2])
        output = self.final_dense(attn, training=training)
        return output, attn


class BaseAlignment(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_classes=37, blank_index=0, *args, **kwargs):
        super(BaseAlignment, self).__init__(*args, **kwargs)
        self.embed_dim   = embed_dim
        self.num_classes = num_classes
        self.blank_index = blank_index

    def build(self, input_shape):
        self.dense1 = Dense(self.embed_dim, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.dense2 = Dense(self.num_classes, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))
        self.get_length = GetTokenLength(self.blank_index)

    def call(self, inputs, training=False):
        l_feature, v_feature = inputs
        f = concatenate(inputs, axis=-1)
        f_att = self.dense1(f, training=training)
        f_att = tf.nn.sigmoid(f_att)
        output = f_att * v_feature + (1 - f_att) * l_feature
        output = self.dense2(output, training=training)
        lengths = self.get_length(output)
        return output, lengths


class ABINet(tf.keras.Model):

    def __init__(self,
                 backbone,
                 transform_net=None,
                 embed_dim=512, 
                 num_heads=8, 
                 out_dim=2048, 
                 loop_iters=3, 
                 encoder_layers=3,
                 decoder_layers=4,
                 max_length=50, 
                 num_classes=37, 
                 blank_index=0, 
                 drop_rate=0.1, 
                 *args, **kwargs):
        super(ABINet, self).__init__(*args, **kwargs)
        self.backbone       = backbone
        self.transform_net  = transform_net
        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.out_dim        = out_dim
        self.loop_iters     = loop_iters
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers  
        self.max_length     = max_length
        self.num_classes    = num_classes
        self.blank_index    = blank_index
        self.drop_rate      = drop_rate

    def build(self, input_shape):
        self.vision = BaseVision(self.embed_dim, 
                                 self.num_heads, 
                                 self.out_dim, 
                                 self.encoder_layers,
                                 self.max_length, 
                                 self.num_classes, 
                                 self.blank_index, 
                                 self.drop_rate)
        self.language = BCNLanguage(self.embed_dim, 
                                    self.num_heads, 
                                    self.out_dim, 
                                    self.num_classes, 
                                    self.max_length, 
                                    num_layers=self.decoder_layers, 
                                    auxiliary_attn=False, 
                                    activation='gelu', 
                                    drop_rate=self.drop_rate)
        self.alignment = BaseAlignment(self.embed_dim, self.num_classes, self.blank_index)

    def call(self, inputs, training=False):
        if self.transform_net is not None:
            inputs = self.transform_net(inputs, training=training)

        vision_results = self.backbone(inputs, training=training)
        vision_results = self.vision(vision_results, training=training)
        feature, attn, lengths = copy.copy(vision_results)
        
        lang_feature = []
        align_feature = []
        for _ in range(self.loop_iters):
            feature = tf.nn.softmax(feature, axis=-1)
            lengths = tf.clip_by_value(lengths, 2, self.max_length)
            feature, lang_attn = self.language([feature, lengths], training=training)
            lang_feature.append(tf.nn.softmax(feature, axis=-1))
            feature, lengths = self.alignment([lang_attn, attn], training=training)
            align_feature.append(tf.nn.softmax(feature, axis=-1))

        return [tf.nn.softmax(vision_results[0], axis=-1), 
                lang_feature, 
                align_feature]
        
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
        losses = losses_prepare(loss_object)
        loss_value = 0
        if losses:
            loss_value += losses(y_true, y_pred, lenghts)        
        return loss_value