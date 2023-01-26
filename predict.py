import cv2
import math
import numpy as np
import tensorflow as tf

from models.vgg_bilstm import VGG_BiLSTM
from models.scene_text_recognition import STR
from models.layers.label_converter import CTCLabelConverter

from utils.files import get_files
from visualizer.visual_image import visual_image
from visualizer.visual_value import tensor_value_info


def resize_with_pad(image, target_size, interpolation=cv2.INTER_NEAREST):
    try:
        h, w, _ = image.shape
    except:
        h, w = image.shape
    ratio = w / float(h)
    if math.ceil(target_size[0] * ratio) > target_size[1]:               
        resized_w = target_size[1]                                     
    else:
        resized_w = math.ceil(target_size[0] * ratio)

    image = cv2.resize(image, (resized_w, target_size[0]), interpolation=interpolation)
    new_h, new_w, _ = image.shape
    pad_image = np.zeros(target_size)
    pad_image[:, :new_w, :] = image

    if target_size[1] != new_w:  # add border Pad
        pad_image[:, new_w:, :] = np.expand_dims(image[:, new_w-1, :], axis=1)
    return pad_image


def preprocess_input(image):
    image /= 255.0
    return image


if __name__ == '__main__':
    image_path = "./images"

    target_shape = cfg.STR_TARGET_SIZE
    
    character    = cfg.DATA_CHARACTER
    
    converter    = CTCLabelConverter(character)
    
    num_class    = len(converter.character)
     
    num_filters  = cfg.STR_FILTERS
    
    hidden_dim   = cfg.STR_HIDDEN_DIMENTION
    
    output_dim   = cfg.STR_OUTPUT_DIMENTION

    architecture = VGG_BiLSTM(str_filters, str_hidden_dim, str_output_dim, num_class)
    
    model = STR(architecture, image_size=target_shape)

    weight_type    = "weights"

    weight_objects = [        
                        {
                            'path': './saved_weights/20230126-003551/best_weights',
                            'stage': 'full',
                            'custom_objects': None
                        }
                     ]

    if weight_type and weight_objects:
        if weight_type == "weights":
            model.load_weights(weight_objects)
        elif weight_type == "models":
            model.load_models(weight_objects)

    images = get_files(image_path, extensions=['png', 'jpg'])
    
    for image in images:
        image = cv2.imread(f"{image_path}/{image}")
        image_data  = resize_with_pad(image, target_shape)
        image_data  = preprocess_input(image_data.astype(np.float32))
        image_data  = np.expand_dims(image_data, axis=0)
        preds, preds_length, preds_max_prob = model.predict(image_data)
        preds_str = converter.decode(preds, preds_length)
        visual_image([image], [str(preds_str)])