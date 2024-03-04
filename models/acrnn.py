import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import AveragePooling2D
from utils.train_processing import losses_prepare


class ACRNN(tf.keras.Model):
    def __init__(self, backbone, attention_net=None, sequence_net=None, transform_net=None, *args, **kwargs):
        super(ACRNN, self).__init__(*args, **kwargs)
        self.backbone      = backbone
        self.attention_net = attention_net
        self.sequence_net  = sequence_net
        self.transform_net = transform_net
        self.max_length = attention_net.batch_max_length if attention_net is not None else 2

    def build(self, input_shape):
        if not isinstance(self.sequence_net, ConvolutionHead):
            reduce_lenght         = self.backbone.output.shape[1]
            self.map_to_sequence  = AveragePooling2D((reduce_lenght, 1))
        self.final_activation = Softmax(axis=-1)
        
    def call(self, inputs, training=False):
        x, text = inputs
        text_shape = tf.shape(text)
        final_text = tf.slice(text, [0, 1], [text_shape[0], text_shape[1] - 1])
        model_text = tf.slice(text, [0, 0], [text_shape[0], text_shape[1] - 1])

        if self.transform_net is not None:
            x = self.transform_net(x, training=training)

        x = self.backbone(x, training=training)
        
        if hasattr(self, 'map_to_sequence'):
            x = self.map_to_sequence(x)
            x = tf.squeeze(x, axis=1)

        if self.sequence_net is not None:
            x = self.sequence_net(x, training=training)

        if self.attention_net is not None:
            x = self.attention_net([x, model_text], training=training)

        x = self.final_activation(x)
        return x, final_text

    @tf.function
    def predict(self, inputs):
        dummy_text     = tf.zeros((tf.shape(inputs)[0], self.max_length + 2))
        y_pred         = self([inputs, dummy_text], training=False)
        preds          = tf.math.argmax(y_pred, axis=2)
        preds_max_prob = tf.reduce_max(y_pred, axis=2)
        batch_length   = tf.cast(tf.shape(y_pred)[0], dtype="int32")
        preds_length   = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        preds_length   = preds_length * tf.ones(shape=(batch_length), dtype="int32")
        return preds, preds_length, preds_max_prob
    
    def calc_loss(self, y_true, y_pred, lenghts, loss_object):
        func_loss = losses_prepare(loss_object)
        loss = 0
        if func_loss:
            loss += func_loss(y_true, y_pred, lenghts)
        return loss