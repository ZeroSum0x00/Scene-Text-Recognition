import cv2
import math
import numpy as np


def resize_with_pad(image, target_size, interpolation=None):
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    ratio = w / float(h)
    if math.ceil(target_size[0] * ratio) > target_size[1]:               
        resized_w = target_size[1]                                     
    else:
        resized_w = math.ceil(target_size[0] * ratio)

    image = cv2.resize(image, (resized_w, target_size[0]), interpolation=interpolation)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    new_h, new_w = target_size[0], resized_w
    pad_image = np.zeros(target_size)    
    pad_image[:, :new_w, :] = image

    if target_size[1] != new_w:  # add border Pad
        pad_image[:, new_w:, :] = np.expand_dims(image[:, new_w-1, :], axis=1)
    return pad_image

def image_preprocessing(image, normalize='sub_divide', target_size=None, interpolation=None):    
    if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
        image = resize_with_pad(image, target_size, interpolation)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if normalize == "sub_divide":
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = np.clip(image, -1, 1)
    elif normalize == "divide":
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.clip(image, 0, 1)
    else:
        image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)
    return image
