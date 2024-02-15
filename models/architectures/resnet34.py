import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from models.layers import PositionalEncoding, TransformerEncoderLayer


def BasicBlock(input_tensor, filters, kernel_size=3, strides=(1, 1), downsaple=False):
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)

    if downsaple:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='VALID', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet34(num_blocks, input_shape=(32, 128, 3)):
    img_input = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(num_blocks[0]):
        downsaple = True if i == 0 else False
        strides = 2 if i == 0 else 1
        x = BasicBlock(x, 32, 3, strides, downsaple)

    for i in range(num_blocks[1]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, 64, 3, 1, downsaple)

    for i in range(num_blocks[2]):
        downsaple = True if i == 0 else False
        strides = 2 if i == 0 else 1
        x = BasicBlock(x, 128, 3, strides, downsaple)

    for i in range(num_blocks[3]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, 256, 3, 1, downsaple)

    for i in range(num_blocks[4]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, 512, 3, 1, downsaple)
    model = Model(img_input, x)
    return model


# class ResTranformer(tf.keras.Model):

#     def __init__(self, num_blocks, embed_dim, num_heads, out_dim=2048, num_layers=3, drop_rate=0.1, *args, **kwargs):
#         super(ResTranformer, self).__init__(*args, **kwargs)
#         self.num_blocks    = num_blocks
#         self.embed_dim     = embed_dim
#         self.num_heads     = num_heads
#         self.out_dim       = out_dim
#         self.num_layers    = num_layers
#         self.drop_rate     = drop_rate
        
#     def build(self, input_shape):
#         self.backbone = CustomResNet(num_blocks=self.num_blocks, input_shape=input_shape[1:])
#         self.pos_encoder = PositionalEncoding(self.embed_dim, max_len=8*32, drop_rate=self.drop_rate)
#         self.transformer_encoder = Sequential([
#             TransformerEncoderLayer(embed_dim=self.embed_dim,
#                                     num_heads=self.num_heads,
#                                     out_dim=self.out_dim,
#                                     drop_rate=self.drop_rate) for i in range(self.num_layers)
#         ])

#     def call(self, inputs, training=False):
#         x = self.backbone(inputs, training=training)
#         h = tf.shape(x)[1]
#         w = tf.shape(x)[2]
#         c = tf.shape(x)[-1]

#         x = tf.reshape(x, [-1, h*w, c])
#         x = tf.transpose(x, [1, 0, 2])
#         x = self.pos_encoder(x, training=training)
#         x = self.transformer_encoder(x, training=training)
#         x = tf.transpose(x, [1, 0, 2])
#         x = tf.reshape(x, [-1, h, w, c])
#         return x