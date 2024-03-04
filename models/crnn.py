import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import AveragePooling2D
from models.layers import ConvolutionHead
from utils.train_processing import losses_prepare


class CRNN(tf.keras.Model):
    def __init__(self, backbone, sequence_net=None, transform_net=None, num_classes=1000, *args, **kwargs):
        super(CRNN, self).__init__(*args, **kwargs)
        self.backbone      = backbone
        self.sequence_net  = sequence_net
        self.transform_net = transform_net
        self.num_classes   = num_classes
        
    def build(self, input_shape):
        if not isinstance(self.sequence_net, ConvolutionHead):
            reduce_lenght         = self.backbone.output.shape[1]
            self.map_to_sequence  = AveragePooling2D((reduce_lenght, 1))
            self.predictor        = Dense(units=self.num_classes)

        self.final_activation = Softmax(axis=-1)

    def call(self, inputs, training=False):
        if self.transform_net is not None:
            inputs = self.transform_net(inputs, training=training)
            
        x = self.backbone(inputs, training=training)

        if hasattr(self, 'map_to_sequence'):
            x = self.map_to_sequence(x)
            x = tf.squeeze(x, axis=1)

        if self.sequence_net is not None:
            x = self.sequence_net(x, training=training)
            
        if hasattr(self, 'predictor'):
            x = self.predictor(x, training=training)
            
        x = self.final_activation(x)
        return x

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