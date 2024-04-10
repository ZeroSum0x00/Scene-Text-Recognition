import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

from models.layers import get_activation_from_name, get_normalizer_from_name, ConvolutionBlock, PositionalEmbedding, MLPBlock, DropPath



class ConvolutionMixer(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3, 3), groups=8, size=(8, 25), *args, **kwargs):
        super(ConvolutionMixer, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups
        self.size = size

    def build(self, input_shape):
        self.local_mixer = Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=(1, 1),
                                  padding="SAME",
                                  groups=self.groups)

    def call(self, inputs, training=False):
        bs = tf.shape(inputs)[0]
        x = tf.reshape(inputs, shape=[-1, self.size[0], self.size[1], self.filters])
        x = self.local_mixer(x, training=training)
        x = tf.reshape(x, shape=[bs, -1, self.filters])
        return x


class ExtractPatches(tf.keras.layers.Layer):

    def __init__(self, embed_dim, patch_size=[4, 4], iter=2,  mode='pope', activation='gelu', normalizer='batch-norm', *args, **kwargs):
        super(ExtractPatches, self).__init__(*args, **kwargs)
        self.embed_dim  = embed_dim
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.iter       = iter
        self.mode       = mode
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        if self.mode == 'pope':
            self.projection = Sequential([
                ConvolutionBlock(filters=self.embed_dim // 2**(self.iter - i - 1),
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 padding='SAME',
                                 groups=1,
                                 use_bias=False,
                                 activation=self.activation,
                                 normalizer=self.normalizer) for i in range(self.iter)
            ])
            self.num_patches = (input_shape[1] // (2 ** self.iter)) * (input_shape[2] // (2 ** self.iter))

        elif self.mode == 'normal':
            self.projection = Conv2D(filters=self.embed_dim,
                                     kernel_size=self.patch_size,
                                     strides=self.patch_size,
                                     padding="valid")
            self.num_patches = input_shape[1] // self.patch_size[0] * input_shape[2] // self.patch_size[1]

        self.reshape = Reshape((-1, self.embed_dim))

    def call(self, inputs, training=False):
        x = self.projection(inputs, training=training)
        x = self.reshape(x)
        return x

        
class Attention(tf.keras.layers.Layer):

    def __init__(self, 
                 embed_dim, 
                 num_heads=8, 
                 mixer='Global', 
                 size=None, 
                 local_kernel=(7, 11), 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0., 
                 *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mixer = mixer
        self.size = size
        self.local_kernel = local_kernel
        self.qkv_bias = qkv_bias
        head_dim = embed_dim // num_heads
        self.qk_scale = qk_scale if qk_scale else head_dim**-0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def build(self, input_shape):
        H, W = self.size
        self.N = H * W
        self.C = self.embed_dim

        if self.mixer == 'Local' and self.size:
            hk, wk = self.local_kernel
            mask = np.ones(shape=[self.N, H + hk - 1, W + wk - 1], dtype=np.int32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //2]
            mask_paddle = mask_paddle.reshape(-1, self.N).astype(bool)
            self.mask = np.where(mask_paddle, -1e30, mask_paddle)
            self.mask = self.mask.reshape(1, 1, self.N, self.N)
            self.mask = tf.convert_to_tensor(self.mask, dtype=tf.float32)
            self.mask = tf.Variable(
                initial_value=self.mask, trainable=False, name=f'attention/mask'
            )

        self.qkv_projector = Dense(units=self.embed_dim * 3, use_bias=self.qkv_bias)
        self.attention_drop = Dropout(rate=self.attn_drop)
        self.projector       = Dense(units=self.embed_dim)
        self.projection_drop = Dropout(rate=self.proj_drop)


    def call(self, inputs, training=False):
        if self.size:
            N = self.N
            C = self.C
        else:
            N = tf.shape(inputs)[1]
            C = tf.shape(inputs)[-1]
        qkv = self.qkv_projector(inputs, training=training)
        qkv = tf.reshape(qkv, shape=[-1, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, perm=[0, 3, 1, 4, 2])
        query, key, value = tf.split(qkv, 3, axis=-1)
        query = tf.squeeze(query, axis=-1) * self.qk_scale
        key = tf.squeeze(key, axis=-1)
        value = tf.squeeze(value, axis=-1)

        attn = tf.matmul(query, key, transpose_b=True)

        if self.mixer == "Local":
            attn += self.mask

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attention_drop(attn, training=training)

        x = tf.matmul(attn, value)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.projector(x, training=training)
        x = self.projection_drop(x, training=training)
        return x


class SubSample(tf.keras.layers.Layer):
    def __init__(self, filters, strides=[2, 1], mode='pool', activation=None, normalizer='layer-norm', *args, **kwargs):
        super(SubSample, self).__init__(*args, **kwargs)
        self.filters = filters
        self.strides = strides
        self.mode = mode
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        if self.mode == 'pool':
            self.avg_pool = AveragePooling2D(pool_size=(3, 5), strides=self.strides, padding='SAME')
            self.max_pool = MaxPooling2D(pool_size=(3, 5), strides=self.strides, padding='SAME')
            self.reshape = Reshape((-1, input_shape[-1]))
            self.project = Dense(units=self.filters)
        else:
            self.project = Conv2D(filters=self.filters,
                                  kernel_size=(3, 3),
                                  strides=self.strides,
                                  padding="SAME")
            self.reshape = Reshape((-1, self.filters))

        self.norm = get_normalizer_from_name(self.normalizer)
        self.activ = get_activation_from_name(self.activation)

    def call(self, inputs, training=False):
        if self.mode == 'pool':
            x1 = self.avg_pool(inputs)
            x2 = self.max_pool(inputs)
            x  = (x1 + x2) * 0.5
            x  = self.reshape(x)
            x  = self.project(x, training=training)
        else:
            x  = self.project(inputs, training=training)
            x  = self.reshape(x)

        x = self.norm(x, training=training)
        x = self.activ(x, training=training)
        return x


class SVTRBlock(tf.keras.layers.Layer):
    def __init__(self,
                 mixer_layer,
                 mlp_dim,
                 use_prenorm=False,
                 activation='gelu',
                 normalizer='layer-norm',
                 proj_drop=0.,
                 drop_path_prob=0.,
                 *args, **kwargs):
        super(SVTRBlock, self).__init__(*args, **kwargs)
        self.mixer_layer    = mixer_layer
        self.mlp_dim        = mlp_dim
        self.use_prenorm    = use_prenorm
        self.activation     = activation
        self.normalizer     = normalizer
        self.proj_drop      = proj_drop
        self.drop_path_prob = drop_path_prob

    def build(self, input_shape):
        self.mlp_block = MLPBlock(self.mlp_dim,
                                  activation=self.activation,
                                  normalizer=self.normalizer,
                                  drop_rate=self.proj_drop)
        self.norm_layer1 = get_normalizer_from_name(self.normalizer)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer)
        self.drop_path = DropPath(self.drop_path_prob if self.drop_path_prob > 0. else 0.)

    def call(self, inputs, training=False):
        if self.use_prenorm:
            x = self.mixer_layer(inputs, training=training)
            x = self.drop_path(x, training=training)
            x += inputs
            x = self.norm_layer1(x, training=training)
            x1 = self.mlp_block(x, training=training)
            x1 = self.drop_path(x1, training=training)
            x += x1
            x = self.norm_layer2(x, training=training)
        else:
            x = self.norm_layer1(inputs, training=training)
            x = self.mixer_layer(x, training=training)
            x = self.drop_path(x, training=training)
            x += inputs
            x1 = self.norm_layer2(x, training=training)
            x1 = self.mlp_block(x1, training=training)
            x1 = self.drop_path(x1, training=training)
            x += x1
        return x


def SVTRNet(num_filters=[64, 128, 256],
            num_blocks=[3, 6, 3],
            num_heads=[2, 4, 8],
            patch_size=[4, 4],
            mlp_ratio=4, 
            mixer=['Local'] * 6 + ['Global'] * 6,
            local_kernel=[7, 11],
            qkv_bias=True,
            qk_scale=None,
            submodule_mode=True,
            max_length=25,
            include_top=True, 
            weights='imagenet',
            input_tensor=None, 
            input_shape=None,
            pooling=None,
            use_prenorm=False,
            activation='gelu',
            normalizer='layer-norm',
            final_activation="hard-swish",
            classes=1000,
            attn_drop=0., 
            proj_drop=0.,
            drop_path_rate=0.1,
            final_drop=0.1):
  
    img_input   = Input(shape=input_shape)
    f0, f1, f2  = num_filters
    h0, h1, h2  = num_heads
    n0, n1, n2  = num_blocks
    height_size = input_shape[0] // patch_size[0]
    width_size  = input_shape[1] // patch_size[1]
                
    x = ExtractPatches(embed_dim=f0, 
                       patch_size=patch_size, 
                       iter=2, 
                       mode='pope',
                       activation=activation,
                       normalizer='batch-norm')(img_input)
    x = PositionalEmbedding()(x)
    x = Dropout(proj_drop)(x)

    dpr = [x for x in np.linspace(0., drop_path_rate, sum(num_blocks))]

    for i in range(n0):
        mixer_mode = mixer[0:n0][i]
        drop_rate = dpr[0:n0][i]

        if mixer_mode == 'Conv':
            mixer_layer = ConvolutionMixer(filters=f0, kernel_size=local_kernel, groups=h0, size=[height_size, width_size])
        else:
            mixer_layer = Attention(embed_dim=f0, num_heads=h0, mixer=mixer_mode, size=[height_size, width_size], local_kernel=local_kernel, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        
        mlp_dim = f0 * mlp_ratio
        x = SVTRBlock(mixer_layer, 
                      mlp_dim, 
                      use_prenorm=use_prenorm,
                      activation=activation,
                      normalizer=normalizer,
                      proj_drop=proj_drop,
                      drop_path_prob=drop_rate)(x)

    if submodule_mode:
        x = tf.reshape(x, shape=[-1, height_size, width_size, f0])
        x = SubSample(f1, 
                      strides=[2, 1],
                      mode=submodule_mode, 
                      normalizer=normalizer)(x)
        height_size = height_size // 2

    for i in range(n1):
        mixer_mode = mixer[n0:n0 + n1][i]
        drop_rate = dpr[n0:n0 + n1][i]

        if mixer_mode == 'Conv':
            mixer_layer = ConvolutionMixer(filters=f1, kernel_size=local_kernel, groups=h1, size=[height_size, width_size])
        else:
            mixer_layer = Attention(embed_dim=f1, num_heads=h1, mixer=mixer_mode, size=[height_size, width_size], local_kernel=local_kernel, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        mlp_dim = f1 * mlp_ratio
        x = SVTRBlock(mixer_layer, 
                      mlp_dim, 
                      use_prenorm=use_prenorm,
                      activation=activation,
                      normalizer=normalizer,
                      proj_drop=proj_drop,
                      drop_path_prob=drop_rate)(x)

    if submodule_mode:
        x = tf.reshape(x, shape=[-1, height_size, width_size, f1])
        x = SubSample(f2, 
                      strides=[2, 1],
                      mode=submodule_mode, 
                      normalizer=normalizer)(x)
        height_size = height_size // 2

    for i in range(n2):
        mixer_mode = mixer[n0 + n1:][i]
        drop_rate = dpr[n0 + n1:][i]

        if mixer_mode == 'Conv':
            mixer_layer = ConvolutionMixer(filters=f2, kernel_size=local_kernel, groups=h2, size=[height_size, width_size])
        else:
            mixer_layer = Attention(embed_dim=f2, num_heads=h2, mixer=mixer_mode, size=[height_size, width_size], local_kernel=local_kernel, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        
        mlp_dim = f2 * mlp_ratio
        x = SVTRBlock(mixer_layer, 
                      mlp_dim, 
                      use_prenorm=use_prenorm,
                      activation=activation,
                      normalizer=normalizer,
                      proj_drop=proj_drop,
                      drop_path_prob=drop_rate)(x)

    if not use_prenorm:
        x = get_normalizer_from_name(normalizer)(x)

    if include_top:
        x = tf.reshape(x, shape=[-1, height_size, width_size, f2])
        x = AveragePooling2D((height_size, int(width_size // max_length) if width_size > max_length else 1))(x)
        x = Conv2D(filters=classes,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding="VALID",
                   use_bias=False)(x)
        x = get_activation_from_name(final_activation)(x)
        x = Dropout(final_drop)(x)

    model = Model(inputs=img_input, outputs=x, name='SVTRNet')
    return model