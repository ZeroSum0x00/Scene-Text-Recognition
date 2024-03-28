import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from .simple_block import ConvolutionBlock


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


class ConvolutionHead(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_classes=1000, activation='relu', normalizer='batch-norm', *args, **kwargs):
        super(ConvolutionHead, self).__init__(*args, **kwargs)
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes
        self.activation  = activation
        self.normalizer  = normalizer

    def build(self, input_shape):
        self.block = Sequential([
            ConvolutionBlock(self.hidden_dim, 1, activation=self.activation, normalizer=self.normalizer),
            ConvolutionBlock(self.num_classes, 1, activation=None, normalizer=None)
        ])

    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        x = tf.squeeze(x, axis=1)
        return x


class SimpleSVTRHead(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=-1, num_classes=1000, *args, **kwargs):
        super(SimpleSVTRHead, self).__init__(*args, **kwargs)
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

    def build(self, input_shape):
        if self.hidden_dim != -1:
            self.block = Sequential([
                Dense(units=self.hidden_dim),
                Dense(units=self.num_classes)
            ])
        else:
            self.block = Sequential([
                Dense(units=self.num_classes)
            ])

    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        return x


class EncodeSVTRHead(tf.keras.Model):
    def __init__(self, 
                 hidden_dim=120, 
                 local_kernel=[7, 11],
                 conv_kernel_size=[3, 3],
                 mixer_mode='Global',
                 depth=2, 
                 num_heads=8, 
                 qkv_bias=True, 
                 qk_scale=None,
                 mlp_ratio=2.0,  
                 activation='swish',
                 conv_normalizer='batch-norm',
                 attn_normalizer='layer-norm',
                 use_prenorm=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path_rate=0.1,
                 num_classes=1000, 
                 *args, **kwargs):
        super(EncodeSVTRHead, self).__init__(*args, **kwargs)
        self.hidden_dim       = hidden_dim
        self.local_kernel     = local_kernel
        self.conv_kernel_size = conv_kernel_size
        self.mixer_mode       = mixer_mode
        self.depth            = depth
        self.num_heads        = num_heads
        self.qkv_bias         = qkv_bias
        self.qk_scale         = qk_scale
        self.mlp_ratio        = mlp_ratio
        self.activation       = activation
        self.conv_normalizer  = conv_normalizer
        self.attn_normalizer  = attn_normalizer
        self.use_prenorm      = use_prenorm
        self.attn_drop        = attn_drop
        self.proj_drop        = proj_drop
        self.drop_path_rate   = drop_path_rate
        self.num_classes      = num_classes

    def build(self, input_shape):
        c = input_shape[-1]
        self.conv1 = ConvolutionBlock(filters=c // 8,
                                      kernel_size=self.conv_kernel_size,
                                      strides=(1, 1),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=self.activation,
                                      normalizer=self.conv_normalizer)
        self.conv2 = ConvolutionBlock(filters=self.hidden_dim,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='VALID',
                                      use_bias=False,
                                      activation=self.activation,
                                      normalizer=self.conv_normalizer)
        self.conv3 = ConvolutionBlock(filters=c,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='VALID',
                                      use_bias=False,
                                      activation=self.activation,
                                      normalizer=self.conv_normalizer)
        self.conv4 = ConvolutionBlock(filters=c // 8,
                                      kernel_size=self.conv_kernel_size,
                                      strides=(1, 1),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=self.activation,
                                      normalizer=self.conv_normalizer)
        self.conv5 = ConvolutionBlock(filters=self.num_classes,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='VALID',
                                      use_bias=False,
                                      activation=self.activation,
                                      normalizer=self.conv_normalizer)
        
        if self.mixer_mode == 'Conv':
            mixer_layer = ConvolutionMixer(filters=self.hidden_dim, kernel_size=self.local_kernel, groups=self.num_heads, size=None)
        else:
            mixer_layer = Attention(embed_dim=self.hidden_dim, 
                                    num_heads=self.num_heads, 
                                    mixer=self.mixer_mode, 
                                    size=None, 
                                    local_kernel=self.local_kernel, 
                                    qkv_bias=self.qkv_bias, 
                                    qk_scale=self.qk_scale, 
                                    attn_drop=self.attn_drop, 
                                    proj_drop=self.proj_drop)

        mlp_dim = self.hidden_dim * self.mlp_ratio
        self.block = Sequential([
            SVTRBlock(mixer_layer,
                      mlp_dim,
                      use_prenorm=self.use_prenorm,
                      activation=self.activation,
                      normalizer=self.attn_normalizer,
                      proj_drop=self.proj_drop,
                      drop_path_prob=self.drop_path_rate) for _ in range(self.depth)
        ])
        self.norm_layer = get_normalizer_from_name(self.attn_normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        bs, h, w, c = tf.shape(x)
        x = tf.reshape(x, shape=[bs, h*w, c])
        x = self.block(x, training=training)
        x = self.norm_layer(x, training=training)
        x = tf.reshape(x, shape=[bs, h, w, c])
        x = self.conv3(x, training=training)
        x = concatenate([x, inputs])
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        return x