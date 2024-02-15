import os
import cv2
import math
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from models import build_models
from utils.files import get_files
from utils.post_processing import image_preprocessing
from utils.config_processing import load_config
from utils.auxiliary_processing import change_color_space


def predict(file_config=None):
    config = load_config(file_config)
    test_config = config['Test']
    data_config  = config['Dataset']
    data_path = test_config['data']
    
    converter, model = build_models(config['Model'], './saved_weights/20240124-225756/weights/best_valid_word_accuracy')
    

    images = get_files(data_path, extensions=['jpg', 'png'])
    target_shape = config['Model']['input_shape']

    total_num = len(images)
    n_correct = 0
    for name in tqdm(images):
        if len(target_shape) == 2 or target_shape[-1] == 1:
            read_mode = 0
        else:
            read_mode = 1

        label_str = name.split('_')[-1].split('.')[0]
        image = cv2.imread(f"{data_path}/{name}", read_mode)
        if read_mode == 1:
            image = change_color_space(image, 'bgr', data_config['data_info']['color_space'])
        image = image_preprocessing(image, 
                                    target_size=target_shape, 
                                    interpolation=cv2.INTER_NEAREST)
        image  = tf.expand_dims(image, axis=0)

        pred, preds_length, pred_max_prob = model.predict(image)

        pred_str = converter.decode(pred, preds_length)

        if label_str in pred_str:
            n_correct += 1
            
    accuracy = n_correct / float(total_num) * 100
    print(f'\nCurrent accucary: {accuracy}% ({n_correct} in {total_num} sample)')

if __name__ == '__main__':
    predict('./configs/None-resnet-bilstm_ctc.yaml')