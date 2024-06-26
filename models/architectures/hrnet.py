import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate

from models.layers import get_activation_from_name, get_normalizer_from_name, ConvolutionBlock


class BasicBlock(tf.keras.layers.Layer):
    
    def __init__(self, 
                 filters, 
                 shortcut=True,
                 activation = 'relu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.filters    = filters
        self.shortcut   = shortcut
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        self.c    = input_shape[-1]
        self.conv = Sequential([
            ConvolutionBlock(self.filters, 3, 1, use_bias=False, activation=self.activation, normalizer=self.normalizer),
            ConvolutionBlock(self.filters, 3, 1, use_bias=False, activation=None, normalizer=self.normalizer)
        ])
        self.final_activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
            
        x = self.final_activ(x, training=training)
        return x


class Bottleneck(tf.keras.Model):
    
    def __init__(self, 
                 filters, 
                 downsample=False,
                 expansion  = 4,
                 shortcut=True,
                 activation = 'relu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.filters    = filters
        self.downsample = downsample
        self.expansion  = expansion
        self.shortcut   = shortcut
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        self.c    = input_shape[-1]
        self.conv = Sequential([
            ConvolutionBlock(self.filters, 1, 1, use_bias=False, activation=self.activation, normalizer=self.normalizer),
            ConvolutionBlock(self.filters, 3, 1, use_bias=False, activation=self.activation, normalizer=self.normalizer),
            ConvolutionBlock(self.filters * self.expansion, 1, 1, activation=None, normalizer=self.normalizer)
        ])
        self.final_activ = get_activation_from_name(self.activation)
        if self.downsample:
            self.down_conv = ConvolutionBlock(self.filters * self.expansion, 1, 1, use_bias=False, activation=None, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        if hasattr(self, 'down_conv'):
            identity = self.down_conv(inputs, training=training)
        else:
            identity = inputs
            
        x = self.conv(inputs, training=training)
            
        if self.shortcut and self.c == self.filters:
            x = add([identity, x])

        x = self.final_activ(x, training=training)
        return x


class UpsampleBlock(tf.keras.layers.Layer):
    
    def __init__(self, 
                 up_factor     = 2,
                 activation    = None, 
                 normalizer    = 'batch-norm', 
                 interpolation = 'bilinear',
                 **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)
        self.up_factor     = up_factor
        self.activation    = activation
        self.normalizer    = normalizer
        self.interpolation = interpolation
                     
    def build(self, input_shape):
        c = input_shape[-1]
        self.block = Sequential([
            UpSampling2D(size=self.up_factor, interpolation=self.interpolation),
            ConvolutionBlock(c//self.up_factor, 1, 1, use_bias=False, activation=self.activation, normalizer=self.normalizer),
        ])
        
    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        return x


class DownsampleBlock(tf.keras.layers.Layer):
    
    def __init__(self, 
                 num_samplings = 2,
                 activation    = 'relu', 
                 normalizer    = 'batch-norm', 
                 **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)
        self.num_samplings = num_samplings
        self.activation    = activation
        self.normalizer    = normalizer
                     
    def build(self, input_shape):
        c = input_shape[-1]
        self.list_block = []
        for i in range(self.num_samplings):
            activ = self.activation if i < self.num_samplings -1 else None
            self.list_block.append(ConvolutionBlock(c*2, 3, 2, use_bias=False, activation=activ, normalizer=self.normalizer))
            c *= 2
        self.block = Sequential(self.list_block)

    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        return x


class HRBlock(tf.keras.Model):
    
    def __init__(self,
                 block,
                 num_block_per_stage = 4,
                 last_stage    = False,
                 activation    = 'relu', 
                 normalizer    = 'batch-norm', 
                 **kwargs):
        super(HRBlock, self).__init__(**kwargs)
        self.block         = block
        self.num_block_per_stage = num_block_per_stage
        self.last_stage    = last_stage 
        self.activation    = activation
        self.normalizer    = normalizer
                     
    def build(self, input_shape):
        self.iters = len(input_shape)
        c = input_shape[0][-1]
        self.activ = get_activation_from_name(self.activation)
        
        self.parallel_conv_lists = []
        for i in range(self.iters):
            c_i = c * 2**i
            block_list = []
            for j in range(self.num_block_per_stage):
                block_list.append(self.block(c_i, activation=self.activation, normalizer=self.normalizer, shortcut=False))
            self.parallel_conv_lists.append(Sequential(block_list))

        self.up_conv_lists = []
        for i in range(self.iters - 1):
            block_list = []
            for j in range(i + 1, self.iters):
                up_factor = 2 ** (j - i)
                block_list.append(UpsampleBlock(up_factor=up_factor, normalizer=self.normalizer))
            self.up_conv_lists.append(block_list)

        self.down_conv_lists = []
        for i in range(1, self.iters if self.last_stage else self.iters + 1):
            block_list = []
            for j in range(i):
                num_samplings = i - j
                block_list.append(DownsampleBlock(num_samplings=num_samplings, activation=self.activation, normalizer=self.normalizer))
            self.down_conv_lists.append(block_list)

    def call(self, inputs, training=False):
        parallel_res_list = []
        for i in range(self.iters):
            x = inputs[i]
            x = self.parallel_conv_lists[i](x, training=training)
            parallel_res_list.append(x)

        final_res_list = []
        for i in range(self.iters if self.last_stage else self.iters + 1):
            # Downsampling all streams to a dimension just lower than the lowest stream, for next stage (Don't do for last stage i.e. index = 4 obviously)
            if i == self.iters:
                x = 0
                for t, m in zip(parallel_res_list, self.down_conv_lists[-1]):
                    x += m(t, training=training)
            else:
                x = parallel_res_list[i]
                # Upsampling all streams (except the uppermost), to all possible dimensions above it till the highest stream
                if i != self.iters - 1:
                    res_list = parallel_res_list[i+1:]
                    up_x = 0
                    for t, m in zip(res_list, self.up_conv_lists[i]):
                        up_x += m(t, training=training)
                    x += up_x
                    
                # Downsampling all streams (except the lowest) to all possible dimensions below it till the lowest stream dimension
                if i != 0:
                    res_list = parallel_res_list[:i]
                    down_x = 0
                    for t, m in zip(res_list, self.down_conv_lists[i - 1]):
                        down_x += m(t, training=training)
                    x += down_x
            x = self.activ(x, training=training)
            final_res_list.append(x)
        return final_res_list


def HRNet_FeatureExtractor(input_shape=(32, 400, 1), filters=32, hidden_dims=64, iters=4, activation='relu', normalizer='batch-norm', classes=1000):
    img_input = Input(shape=input_shape)
    x = ConvolutionBlock(64, 1, 1, use_bias=False, activation=activation, normalizer=normalizer)(img_input)
    x = ConvolutionBlock(64, 1, 1, use_bias=False, activation=activation, normalizer=normalizer)(x)

    for i in range(4):
        downsample = True if i == 0 else False
        x = Bottleneck(64,
                       downsample = downsample,
                       expansion  = 4,
                       activation = activation, 
                       normalizer = normalizer)(x)

    x1 = ConvolutionBlock(hidden_dims, 1, 1, use_bias=False, activation=activation, normalizer=normalizer)(x)
    x2 = ConvolutionBlock(hidden_dims * 2, 3, 2, use_bias=False, activation=activation, normalizer=normalizer)(x)
    x_list = [x1, x2]
    for i in range(iters - 1):
        x_list = HRBlock(BasicBlock,
                         num_block_per_stage = 4,
                         last_stage    = True if i == iters - 2 else False,
                         activation    = activation, 
                         normalizer    = normalizer)(x_list)
        
    res_list = [x_list[0]]
    for i, t in enumerate(x_list[1:]):
        up_factor = 2 ** (i + 1)
        res_list.append(UpSampling2D(size=up_factor, interpolation='bilinear')(t))
        
    x = concatenate(res_list, axis=-1)
    x = ConvolutionBlock(hidden_dims * (1 + 2 + 4 + 8), 1, 1, use_bias=False, activation=activation, normalizer=normalizer)(x)
    x = Conv2D(filters=filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="VALID")(x)
    return Model(img_input, x, name='HRNet')