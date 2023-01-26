import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense


class VGG_FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, num_filters, *args, **kwargs):
        super(VGG_FeatureExtractor, self).__init__(*args, **kwargs)
        self.num_filters = num_filters
    
    def build(self, input_shape):
        f0, f1, f2, f3 = self.num_filters
        self.block0 = Sequential([
                        Conv2D(filters=f0, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                        Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                        Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),

                        Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False),
                        BatchNormalization(),
                        Activation('relu'),
                        Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False),
                        BatchNormalization(),
                        Activation('relu'),
                        MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
                        Conv2D(filters=f3, kernel_size=(2, 2), strides=(1, 1), padding='valid'),
                        Activation('relu'),
        ])

    def call(self, inputs, training=False):
        x = self.block0(inputs, training=training)
        return x


class BidirectionalLSTM(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, *args, **kwargs):
        super(BidirectionalLSTM, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.rnn = Bidirectional(LSTM(units=self.hidden_dim, return_sequences=True), input_shape=input_shape)

    def call(self, inputs, training=False):
        x = self.rnn(inputs)
        return x


class VGG_BiLSTM(tf.keras.Model):
    def __init__(self, num_filters, hidden_dim, output_dim, n_classes, *args, **kwargs):
        super(VGG_BiLSTM, self).__init__(*args, **kwargs)
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_classes = n_classes
        
    def build(self, input_shape):
        self.feature_extractor = VGG_FeatureExtractor(self.num_filters)
        self.map_to_sequence = Reshape(target_shape=(-1, self.num_filters[-1]))
        self.sequence_modeling = BidirectionalLSTM(self.hidden_dim)
        # self.sequence_modeling2 = BidirectionalLSTM(self.hidden_dim)
        self.predictor = Dense(units=self.n_classes)
        self.final_activation = tf.keras.layers.Softmax()
        
    def call(self, inputs, training=False):
        x = self.feature_extractor(inputs, training=training)
        # x = tf.squeeze(x, axis=1)
        x = self.map_to_sequence(x)
        x = self.sequence_modeling(x)
        x = self.sequence_modeling(x)
        x = self.predictor(x)
        # x = -tf.nn.log_softmax(x, axis=-1)
        x = self.final_activation(x)
        return x

    @tf.function
    def predict(self, inputs):
        y_pred = self(inputs, training=False)
        preds_max_prob = tf.reduce_max(y_pred, axis=2)
        batch_length = tf.cast(tf.shape(y_pred)[0], dtype="int32")
        preds_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        preds_length = preds_length * tf.ones(shape=(batch_length), dtype="int32")
        preds = tf.math.argmax(y_pred, axis=2)
        return preds, preds_length, preds_max_prob