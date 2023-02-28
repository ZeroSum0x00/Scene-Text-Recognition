import cv2
import math
import numpy as np
import tensorflow as tf

from models import STR, CRNN, CTCLabelConverter, VGG
from utils.files import get_files
from utils.post_processing import image_preprocessing


if __name__ == '__main__':
    image_path = "/home/vbpo/Desktop/TuNIT/working/Datasets/LP6/validation/"

    target_shape = [32, 200, 1]

    character    = cfg.DATA_CHARACTER

    converter    = CTCLabelConverter(character)

    num_class    = len(converter.character)

    num_filters  = cfg.STR_FILTERS

    hidden_dim   = cfg.STR_HIDDEN_DIMENTION


    backbone = VGG(num_filters)

    architecture = CRNN(backbone, num_filters, hidden_dim, num_class)

    model = STR(architecture, image_size=target_shape)        

    weight_type    = "weights"

    weight_objects = [        
                        {
                            'path': './saved_weights/20230222-134031/best_validation_accuracy',
                            # 'path': './saved_weights/20230223-083045/best_validation_accuracy',
                            'stage': 'full',
                            'custom_objects': None
                        }
                     ]

    if weight_type and weight_objects:
        if weight_type == "weights":
            model.load_weights(weight_objects)
        elif weight_type == "models":
            model.load_models(weight_objects)

    images = get_files(image_path, extensions=['jpg'])

    total_num = len(images)
    n_correct = 0
    for name in tqdm(images):
        if len(target_shape) == 2 or target_shape[-1] == 1:
            read_mode = 0
        else:
            read_mode = 1
        label_str = name.split('_')[-1].split('.')[0]
        image = cv2.imread(f"{image_path}/{name}", read_mode)
        image = image_preprocessing(image, 
                                    target_size=target_shape, 
                                    interpolation=cv2.INTER_NEAREST)
        image  = tf.expand_dims(image, axis=0)

        pred, preds_length, pred_max_prob = model.predict(image)
        pred_str = converter.decode(pred, preds_length)

        if label_str == pred_str:
            n_correct += 1
            
    accuracy = n_correct / float(total_num) * 100
    print(f'\nCurrent accucary: {accuracy}% ({n_correct} in {total_num} sample)')
