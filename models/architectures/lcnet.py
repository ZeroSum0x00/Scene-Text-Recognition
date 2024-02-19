import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from models.layers import ConvolutionBlock, RepVGGBlock


class SEBlock(tf.keras.layers.Layer):
    
    def __init__(self, 
                 expansion  = 0.5,
                 activation = 'relu', 
                 normalizer = None, 
                 *args, 
                 **kwargs):
        super(SEBlock, self).__init__(*args, **kwargs)
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        
    def build(self, input_shape):
        bs = input_shape[-1]
        hidden_dim = int(bs * self.expansion)
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(bs, 1, activation='hard-sigmoid', normalizer=self.normalizer)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.avg_pool(inputs)
        x = self.conv1(x, training=training)        
        x = self.conv2(x, training=training)
        return x * inputs

        
class DepthwiseSeparable(tf.keras.layers.Layer):
    
    def __init__(self, 
                 filters,
                 dw_kernel  = (3, 3),
                 strides    = (1, 1),
                 padding    = "SAME",
                 groups     = 1,
                 expansion  = 1,
                 use_se     = False,
                 activation = 'hard-swish', 
                 normalizer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(DepthwiseSeparable, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.dw_kernel  = dw_kernel
        self.expansion  = expansion
        self.strides    = strides
        self.padding    = padding
        self.groups     = groups
        self.use_se     = use_se
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        if isinstance(self.filters, (list, tuple)):
            f0, f1 = self.filters
        else:
            f0, f1 = self.filters
        self.depthwise_conv = ConvolutionBlock(filters=int(f0 * self.expansion),
                                               kernel_size=self.dw_kernel,
                                               strides=self.strides,
                                               padding=self.padding,
                                               groups=int(self.groups * self.expansion),
                                               activation=self.activation,
                                               normalizer=self.normalizer)
        self.pointwise_conv = ConvolutionBlock(filters=int(f1 * self.expansion),
                                               kernel_size=1,
                                               strides=(1, 1),
                                               padding='VALID')
        if self.use_se:
            self.se_block = SEBlock(expansion=0.25)

    def call(self, inputs, training=False):
        x = self.depthwise_conv(inputs, training=training)
        if hasattr(self, 'se_block'):
            x = self.se_block(x, training=training)
        x = self.pointwise_conv(x, training=training)
        return x


class RepDepthwiseSeparable(DepthwiseSeparable):

    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(self.filters, (list, tuple)):
            f0, f1 = self.filters
        else:
            f0, f1 = self.filters
        self.depthwise_conv = RepVGGBlock(filters=int(f0 * self.expansion),
                                          kernel_size=self.dw_kernel,
                                          strides=self.strides,
                                          padding=self.padding,
                                          groups=int(self.groups * self.expansion),
                                          activation=self.activation,
                                          normalizer=self.normalizer,
                                          training=False)

        
def LCNet(num_filters, input_shape=(32, 200, 3), expansion=0.5, activation='hard-swish', normalizer='batch-norm'):
    img_input = Input(shape=input_shape)
    f0, f1, f2, f3, f4, f5 = num_filters

    x = ConvolutionBlock(filters=int(f0 * expansion),
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         padding="SAME",
                         activation=activation,
                         normalizer=normalizer)(img_input)

    x = DepthwiseSeparable(filters=[f0, f1],
                           dw_kernel=(3, 3),
                           strides=(1, 1),
                           padding="SAME",
                           groups=f0,
                           expansion=expansion,
                           activation=activation,
                           normalizer=normalizer)(x)

    x = DepthwiseSeparable(filters=[f1, f2],
                           dw_kernel=(3, 3),
                           strides=(1, 1),
                           padding="SAME",
                           groups=f1,
                           expansion=expansion,
                           activation=activation,
                           normalizer=normalizer)(x)

    x = DepthwiseSeparable(filters=[f2, f2],
                           dw_kernel=(3, 3),
                           strides=(1, 1),
                           padding="SAME",
                           groups=f2,
                           expansion=expansion,
                           activation=activation,
                           normalizer=normalizer)(x)

    x = DepthwiseSeparable(filters=[f2, f3],
                           dw_kernel=(3, 3),
                           strides=(2, 1),
                           padding="SAME",
                           groups=f2,
                           expansion=expansion,
                           activation=activation,
                           normalizer=normalizer)(x)

    x = DepthwiseSeparable(filters=[f3, f3],
                           dw_kernel=(3, 3),
                           strides=(1, 1),
                           padding="SAME",
                           groups=f3,
                           expansion=expansion,
                           activation=activation,
                           normalizer=normalizer)(x)

    x = DepthwiseSeparable(filters=[f3, f4],
                           dw_kernel=(3, 3),
                           strides=(2, 1),
                           padding="SAME",
                           groups=f3,
                           expansion=expansion,
                           activation=activation,
                           normalizer=normalizer)(x)

    for _ in range(5):
        x = DepthwiseSeparable(filters=[f4, f4],
                               dw_kernel=(5, 5),
                               strides=(1, 1),
                               padding="SAME",
                               groups=f4,
                               expansion=expansion,
                               activation=activation,
                               normalizer=normalizer)(x)

    x = DepthwiseSeparable(filters=[f4, f5],
                           dw_kernel=(5, 5),
                           strides=(2, 1),
                           padding="SAME",
                           groups=f4,
                           expansion=expansion,
                           use_se=True,
                           activation=activation,
                           normalizer=normalizer)(x)

    x = DepthwiseSeparable(filters=[f5, f5],
                           dw_kernel=(5, 5),
                           strides=(1, 1),
                           padding="SAME",
                           groups=f5,
                           expansion=expansion,
                           use_se=True,
                           activation=activation,
                           normalizer=normalizer)(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    model = Model(inputs=img_input, outputs=x, name='LCNet')
    return model


def RepLCNet(num_filters, input_shape=(32, 200, 3), expansion=0.5, activation='hard-swish', normalizer='batch-norm'):
    img_input = Input(shape=input_shape)
    f0, f1, f2, f3, f4, f5 = num_filters

    x = ConvolutionBlock(filters=int(f0 * expansion),
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         padding="SAME",
                         activation=activation,
                         normalizer=normalizer)(img_input)

    x = RepDepthwiseSeparable(filters=[f0, f1],
                              dw_kernel=(3, 3),
                              strides=(1, 1),
                              padding="SAME",
                              groups=f0,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    x = RepDepthwiseSeparable(filters=[f1, f2],
                              dw_kernel=(3, 3),
                              strides=(1, 1),
                              padding="SAME",
                              groups=f1,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    x = RepDepthwiseSeparable(filters=[f2, f2],
                              dw_kernel=(3, 3),
                              strides=(1, 1),
                              padding="SAME",
                              groups=f2,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    x = RepDepthwiseSeparable(filters=[f2, f3],
                              dw_kernel=(3, 3),
                              strides=(2, 1),
                              padding="SAME",
                              groups=f2,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    x = RepDepthwiseSeparable(filters=[f3, f3],
                              dw_kernel=(3, 3),
                              strides=(1, 1),
                              padding="SAME",
                              groups=f3,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    x = RepDepthwiseSeparable(filters=[f3, f4],
                              dw_kernel=(3, 3),
                              strides=(2, 1),
                              padding="SAME",
                              groups=f3,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    for _ in range(5):
        x = RepDepthwiseSeparable(filters=[f4, f4],
                                  dw_kernel=(5, 5),
                                  strides=(1, 1),
                                  padding="SAME",
                                  groups=f4,
                                  expansion=expansion,
                                  activation=activation,
                                  normalizer=normalizer)(x)

    x = RepDepthwiseSeparable(filters=[f4, f5],
                              dw_kernel=(5, 5),
                              strides=(2, 1),
                              padding="SAME",
                              groups=f4,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    x = RepDepthwiseSeparable(filters=[f5, f5],
                              dw_kernel=(5, 5),
                              strides=(1, 1),
                              padding="SAME",
                              groups=f5,
                              expansion=expansion,
                              activation=activation,
                              normalizer=normalizer)(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    model = Model(inputs=img_input, outputs=x, name='LCNet')
    return model