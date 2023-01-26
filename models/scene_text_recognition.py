import copy
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.logger import logger


class STR(tf.keras.Model):
    def __init__(self, 
                 architecture,
                 image_size=(32, 200, 3)):
        super(STR, self).__init__()
        self.architecture = architecture
        self.image_size = image_size
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super(STR, self).compile()
        self.optimizer = optimizer
        self.str_loss = loss
        self.list_metrics = metrics

    @property
    def metrics(self):
        if self.list_metrics:
            return [
                self.total_loss_tracker,
                *self.list_metrics,
            ]
        else:
            return [self.total_loss_tracker]

    def train_step(self, data):
        images, labels, lenghts = data

        with tf.GradientTape() as tape:
            y_pred = self.architecture(images, training=True)
            loss_value = self.str_loss(labels, y_pred, lenghts)

        gradients = tape.gradient(loss_value, self.architecture.trainable_variables)
        # Same torch.nn.utils.clip_grad_norm_
        # https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))

        self.total_loss_tracker.update_state(loss_value)
        if self.list_metrics:
            for metric in self.list_metrics:
                metric.reset_state()
                metric.update_state(labels, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results
    
    def test_step(self, data):
        images, labels, lenghts = data
        y_pred = self.architecture(images, training=False)
        loss_value = self.str_loss(labels, y_pred, lenghts)
        self.total_loss_tracker.update_state(loss_value)
        if self.list_metrics:
            for metric in self.list_metrics:
                metric.reset_state()
                metric.update_state(labels, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        try:
            pred = self.predict(inputs)
            return pred
        except:
            return inputs

    @tf.function
    def predict(self, inputs):
        pred = self.architecture.predict(inputs)
        return pred

    def save_weights(self, weight_path, save_head=True, save_format='tf', **kwargs):
        if save_head:
            self.architecture.save_weights(weight_path, save_format=save_format, **kwargs)
        # else:
        #     backup_model = copy.deepcopy(self.encoder)
        #     backup_model.get_layer("medium_bbox_predictor").pop()
        #     backup_model.get_layer("large_bbox_predictor").pop()
        #     backup_model.get_layer("small_bbox_predictor").pop()
        #     backup_model.save_weights(weight_path, save_format=save_format, **kwargs)

    def load_weights(self, weight_objects):
        for weight in weight_objects:
            weight_path = weight['path']
            custom_objects = weight['custom_objects']
            if weight_path:
                # self.architecture.build(input_shape=self.image_size)
                # self.architecture.built = True
                self.architecture.load_weights(weight_path)
                logger.info("Load STR weights from {}".format(weight_path))

    def save_models(self, weight_path, save_format='tf'):
        self.architecture.save(weight_path, save_format=save_format)

    def load_models(self, weight_objects):
        for weight in weight_objects:
            weight_path = weight['path']
            custom_objects = weight['custom_objects']
            if weight_path:
                self.architecture = load_model(weight_path, custom_objects=custom_objects)
                logger.info("Load STR model from {}".format(weight_path))

    def get_config(self):
        config = super().get_config()
        config.update({
                "architecture": self.architecture,
                "total_loss_tracker": self.total_loss_tracker,
                "optimizer": self.optimizer
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)