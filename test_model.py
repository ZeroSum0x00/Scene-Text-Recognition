import numpy as np
import tensorflow as tf
# from models import ABINet



x = np.random.randint(0, 255, (8, 32, 128, 3))
x = x / 255.0
x = tf.Variable(x, dtype=tf.float32)

# model = ABINet([3, 4, 6, 6, 3], 512, 8, 2048)
# r = model(x)
# print(r)


from models import build_models
from utils.config_processing import load_config


file_config = './configs/crnn/none-vgg-bilstm.yaml'
config = load_config(file_config)
converter, model = build_models(config['Model'])

# print(model.architecture(x))