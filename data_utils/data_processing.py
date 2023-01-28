import os
import cv2
import math
import numpy as np
from glob import glob
from utils.files import extract_zip, verify_folder
from data_utils import ParseTXT, ParseJSON
 

def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = verify_folder(data_destination) + folder_name 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        return data_destination
    else:
        return data_dir


def get_data(data_dir, annotation_dir, 
             character, data_type='TXT', 
             max_string_length=None, sensitive=False, 
             phase='train', check_data=False, load_memory=False,
             *args, **kwargs):
    data_dir = verify_folder(data_dir) + phase
    data_extraction = []

    if data_type.lower() == "text" or data_type.lower() == "txt":
        annotation_file = verify_folder(annotation_dir) + f'{phase}.txt'
        parser = ParseTXT(data_dir, annotation_file, character, max_string_length, sensitive, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser()
    elif data_type.lower() == 'json':
        annotation_file = verify_folder(annotation_dir) + f'{phase}.json'
        parser = ParseJSON(data_dir, annotation_file, character, max_string_length, sensitive, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser()
    dict_data = {
        'data_path': verify_folder(data_dir),
        'data_extractor': data_extraction
    }
    return dict_data


class Normalizer():
    def __init__(self, mode="divide"):
        self.mode = mode

    @classmethod
    def __get_standard_deviation(cls, image, mean=None, std=None):
        if mean:
            for i in range(image.shape[-1]):
                if isinstance(mean, float) or isinstance(mean, int):
                    image[..., i] -= mean
                else:
                    image[..., i] -= mean[i]

        if std:
            for i in range(image.shape[-1]):
                if isinstance(std, float) or isinstance(std, int):
                    image[..., i] /= (std + 1e-20)
                else:
                    image[..., i] /= (std[i] + 1e-20)
        return image

    @classmethod
    def __resize_with_pad(cls, image, target_size, interpolation=None):
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

    def _sub_divide(self, image, mean=None, std=None, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = self.__resize_with_pad(image, target_size, interpolation)
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        if mean or std:
            image = self.__get_standard_deviation(image, mean, std)
        else:
            image = np.clip(image, -1, 1)
        return image

    def _divide(self, image, mean=None, std=None, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = self.__resize_with_pad(image, target_size, interpolation)
        image = image.astype(np.float32)
        image = image / 255.0
        if mean or std:
            image = self.__get_standard_deviation(image, mean, std)
        else:
            image = np.clip(image, 0, 1)
        return image

    def _basic(self, image, mean=None, std=None, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image, mask, bboxes = self.__resize_basic_mode(image, mask, bboxes, target_size, interpolation)
        image = image.astype(np.float32)
        if mean or std:
            image = self.__get_standard_deviation(image, mean, std)
        else:    
            image = image.astype(np.uint8)
            image = np.clip(image, 0, 255)
        return image

    def __call__(self, input, *args, **kargs):
        if self.mode == "divide":
            return self._divide(input, *args, **kargs)
        elif self.mode == "sub_divide":
            return self._sub_divide(input, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)
