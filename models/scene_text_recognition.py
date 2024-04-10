import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.logger import logger


class STR(tf.keras.Model):
    def __init__(self, 
                 architecture):
        super(STR, self).__init__()
        self.architecture = architecture
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super(STR, self).compile()
        self.optimizer = optimizer
        self.loss_object = loss
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
            loss_value = 0
            for losses in self.loss_object:
                if losses['loss'].invariant_name.lower() == "attention_entropy":
                    y_pred, labels = self.architecture([images, labels], training=True)
                else:
                    y_pred = self.architecture(images, training=True)

                loss_value += self.architecture.calc_loss(y_true=labels, 
                                                          y_pred=y_pred, 
                                                          lenghts=lenghts,
                                                          loss_object=losses)
            
        gradients = tape.gradient(loss_value, self.architecture.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.total_loss_tracker.update_state(loss_value)

        if self.list_metrics:
            for metric in self.list_metrics:
                metric.update_state(labels, y_pred, lenghts)
        results = {}
        for metric in self.metrics:
            if isinstance(metric.result(), dict):
                for k, v in metric.result().items():
                    results[k] = v
            else:
                results[metric.name] = metric.result()
        results['learning_rate'] = self.optimizer.lr
        return results
    
    def test_step(self, data):
        images, labels, lenghts = data
        loss_value = 0
        for losses in self.loss_object:
            if losses['loss'].invariant_name.lower() == "attention_entropy":
                y_pred, labels = self.architecture([images, labels], training=False)
            else:
                y_pred = self.architecture(images, training=False)
                
            loss_value += self.architecture.calc_loss(y_true=labels, 
                                                      y_pred=y_pred, 
                                                      lenghts=lenghts, 
                                                      loss_object=losses)
        
        self.total_loss_tracker.update_state(loss_value)
        
        if self.list_metrics:
            for metric in self.list_metrics:
                metric.update_state(labels, y_pred, lenghts)
        results = {}
        for metric in self.metrics:
            if isinstance(metric.result(), dict):
                for k, v in metric.result().items():
                    results[k] = v
            else:
                results[metric.name] = metric.result()
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
        if save_format == "h5" and not weight_path.endswith('.h5'):
            weight_path += '.h5'

        if save_head:
            self.architecture.save_weights(weight_path, save_format=save_format, **kwargs)

    def load_weights(self, weights):
        # self.architecture.build(input_shape=self.image_size)
        # self.architecture.built = True
        self.architecture.load_weights(weights).expect_partial()
        logger.info("Load STR weights from {}".format(weights))

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