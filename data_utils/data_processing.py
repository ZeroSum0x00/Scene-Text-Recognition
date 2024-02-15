import os
import cv2
import math
import numpy as np
from glob import glob
from utils.files import extract_zip, get_files
from data_utils import ParseTXT, ParseJSON, ParseImageName, ParseLMDB
 

def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = os.path.join(data_destination, folder_name)

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        return data_destination
    else:
        return data_dir


def get_labels(label_object):
    if os.path.isfile(label_object):
        with open(label_object, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = ''.join([c.strip() for c in class_names])
        return class_names
    else:
        return label_object

        
def get_data(data_dirs, annotation_dirs, 
             character, data_type='TXT', 
             max_string_length=None, sensitive=False, 
             phase='train', check_data=False, load_memory=False,
             *args, **kwargs):

    def load_data(data_dir, annotation_dir):
        if data_type.lower() == "text" or data_type.lower() == "txt":
            annotation_file = os.path.join(annotation_dir, f'{phase}.txt')
            parser = ParseTXT(data_dir, annotation_file, character, max_string_length, sensitive, load_memory, check_data=check_data, *args, **kwargs)
        elif data_type.lower() == 'json':
            annotation_file = os.path.join(annotation_dir, f'{phase}.json')
            parser = ParseJSON(data_dir, annotation_file, character, max_string_length, sensitive, load_memory, check_data=check_data, *args, **kwargs)
        elif data_type.lower() == 'image' or data_type.lower() == "filename":
            image_files = sorted(get_files(data_dir, extensions=['jpg', 'jpeg', 'png']))
            parser = ParseImageName(data_dir, character, max_string_length, sensitive, load_memory, check_data=check_data, *args, **kwargs)
            return parser(image_files)
        elif data_type.lower() == 'lmdb':
            parser = ParseLMDB(data_dir, character, max_string_length, sensitive, check_data=check_data, *args, **kwargs)
        return parser()

    assert data_type.lower() in ('txt', 'text', 'json', 'image', 'filename', 'lmdb')
    data_extraction = []
    
    if isinstance(data_dirs, (list, tuple)):
        annotation_dirs = annotation_dirs if annotation_dirs else [annotation_dirs] * len(data_dirs)
        for data_dir, annotation_dir in zip(data_dirs, annotation_dirs):
            data_dir = os.path.join(data_dir, phase)
            parser = load_data(data_dir, annotation_dir)
            data_extraction.extend(parser)
    else:
        data_dir = os.path.join(data_dirs, phase)
        parser = load_data(data_dir, annotation_dirs)
        data_extraction.extend(parser)

    return data_extraction


class Normalizer():
    def __init__(self, norm_type="divide", mean=None, std=None, resize_with_pad=True):
        self.norm_type = norm_type
        self.mean      = mean
        self.std       = std
        self.resize_with_pad = resize_with_pad

    def __get_standard_deviation(self, img):
        if self.mean is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.mean, float) or isinstance(self.mean, int):
                    img[..., i] -= self.mean
                else:
                    img[..., i] -= self.mean[i]

        if self.std is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.std, float) or isinstance(self.std, int):
                    img[..., i] /= (self.std + 1e-20)
                else:
                    img[..., i] /= (self.std[i] + 1e-20)
        return img

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

    def _sub_divide(self, image, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            if self.resize_with_pad:
                image = self.__resize_with_pad(image, target_size, interpolation)
            else:
                image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = self.__get_standard_deviation(image)
        image = np.clip(image, -1, 1)
        return image

    def _divide(self, image, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            if self.resize_with_pad:
                image = self.__resize_with_pad(image, target_size, interpolation)
            else:
                image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        image = image / 255.0
        image = self.__get_standard_deviation(image)
        image = np.clip(image, 0, 1)
        return image

    def _basic(self, image, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            if self.resize_with_pad:
                image = self.__resize_with_pad(image, target_size, interpolation)
            else:
                image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        image = self.__get_standard_deviation(image)
        image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)
        return image

    def __call__(self, input, *args, **kargs):
        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                return self._divide(input, *args, **kargs)
            elif self.norm_type == "sub_divide":
                return self._sub_divide(input, *args, **kargs)
        elif isinstance(self.norm_type, types.FunctionType):
            return self._func_calc(input, self.norm_type, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)