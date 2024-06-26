import os
import re
import cv2
import random
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence

from augmenter import build_augmenter
from data_utils.data_augmentation import Augmentor
from data_utils.data_processing import extract_data_folder, get_labels, get_data, Normalizer
from utils.auxiliary_processing import random_range, change_color_space
from utils.logger import logger


def get_train_test_data(data_dirs,
                        annotation_dirs,
                        target_size,
                        batch_size,
                        character,
                        character_converter,
                        max_string_length,
                        sensitive,
                        color_space='RGB',
                        augmentor=None,
                        normalizer='divide',
                        mean_norm=None,
                        std_norm=None,
                        norm_resize_with_pad=False,
                        norm_padding_mode='constant', 
                        norm_padding_color=[255, 255, 255],
                        data_type='txt',
                        check_data=False,
                        load_memory=False,
                        dataloader_mode=0,
                        *args, **kwargs):
    """
        dataloader_mode = 0:   train - validation - test
        dataloader_mode = 1:   train - validation
        dataloader_mode = 2:   train
    """
    character = get_labels(character)
                            
    data_train = get_data(data_dirs, 
                          annotation_dirs   = annotation_dirs,
                          character         = character,
                          data_type         = data_type,
                          max_string_length = max_string_length,
                          sensitive         = sensitive,
                          phase             = 'train',
                          check_data        = check_data,
                          load_memory       = load_memory)
                            
    train_generator = Data_Sequence(data_train, 
                                    target_size             = target_size, 
                                    batch_size              = batch_size, 
                                    character               = character,
                                    character_converter     = character_converter,
                                    color_space             = color_space,
                                    augmentor               = augmentor,
                                    normalizer              = normalizer,
                                    mean_norm               = mean_norm,
                                    std_norm                = std_norm,
                                    norm_resize_with_pad    = norm_resize_with_pad,
                                    norm_padding_mode       = norm_padding_mode, 
                                    norm_padding_color      = norm_padding_color,
                                    phase                   = 'train',
                                    *args, **kwargs)

    if dataloader_mode != 2:
        data_valid = get_data(data_dirs,
                              annotation_dirs   = annotation_dirs,
                              character         = character,
                              data_type         = data_type,
                              max_string_length = max_string_length,
                              sensitive         = sensitive,
                              phase             = 'validation',
                              check_data        = check_data,
                              load_memory       = load_memory)
        valid_generator = Data_Sequence(data_valid, 
                                        target_size             = target_size, 
                                        batch_size              = batch_size, 
                                        character               = character,
                                        character_converter     = character_converter,
                                        color_space             = color_space,
                                        augmentor               = augmentor,
                                        normalizer              = normalizer,
                                        mean_norm               = mean_norm,
                                        std_norm                = std_norm,
                                        norm_resize_with_pad    = norm_resize_with_pad,
                                        norm_padding_mode       = norm_padding_mode, 
                                        norm_padding_color      = norm_padding_color,
                                        phase                   = 'valid',
                                        *args, **kwargs)
    else:
        valid_generator = None

    if dataloader_mode == 1:
        data_test  = get_data(data_dirs,
                              annotation_dirs   = annotation_dirs,
                              character         = character,
                              data_type         = data_type,
                              max_string_length = max_string_length,
                              sensitive         = sensitive,
                              phase             = 'test',
                              check_data        = check_data,
                              load_memory       = load_memory)
        test_generator  = Data_Sequence(data_valid, 
                                        target_size             = target_size, 
                                        batch_size              = batch_size, 
                                        character               = character,
                                        character_converter     = character_converter,
                                        color_space             = color_space,
                                        augmentor               = augmentor,
                                        normalizer              = normalizer,
                                        mean_norm               = mean_norm,
                                        std_norm                = std_norm,
                                        norm_resize_with_pad    = norm_resize_with_pad,
                                        norm_padding_mode       = norm_padding_mode, 
                                        norm_padding_color      = norm_padding_color,
                                        phase                   = 'test',
                                        *args, **kwargs)
    else:
        test_generator = None
        
    logger.info('Load data successfully')
    return train_generator, valid_generator, test_generator


class Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size, 
                 batch_size, 
                 character, 
                 character_converter,
                 color_space='RGB',
                 augmentor=None, 
                 normalizer=None,
                 mean_norm=None, 
                 std_norm=None, 
                 norm_resize_with_pad=False,
                 norm_padding_mode='constant', 
                 norm_padding_color=[255, 255, 255],
                 phase='train', 
                 debug_mode=False):
        self.dataset = dataset
        self.target_size = target_size
        self.batch_size = batch_size
        self.character = character

        if phase == "train":
            self.dataset = shuffle(self.dataset)
        self.N = self.n = len(self.dataset)
                     
        self.color_space = color_space
        self.phase       = phase
        self.debug_mode  = debug_mode
                     
        self.normalizer = Normalizer(normalizer, 
                                     mean=mean_norm, 
                                     std=std_norm, 
                                     resize_with_pad=norm_resize_with_pad, 
                                     padding_mode=norm_padding_mode, 
                                     padding_color=norm_padding_color)
        self.label_converter = character_converter

        if augmentor[phase] and isinstance(augmentor[phase], (tuple, list)):
            self.augmentor = Augmentor(augment_objects=build_augmenter(augmentor[phase]))
        else:
            self.augmentor = augmentor[phase]
                     
    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image = []
        debug_image = []
        batch_label = []
        debug_label = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i           = i % self.N
            sample = self.dataset[i]

            if len(self.target_size) == 2 or self.target_size[-1] == 1:
                deep_channel = 0
            else:
                deep_channel = 1

            if sample['image'] is not None:
                image = sample['image']
                image = change_color_space(image, 'bgr', self.color_space if deep_channel else 'gray')
            else:
                img_path = os.path.join(sample['path'], sample['filename'])
                image = cv2.imread(img_path, deep_channel)
                image = change_color_space(image, 'bgr' if deep_channel else 'gray', self.color_space)

            if self.augmentor:
                image = self.augmentor(image)
                
            image = self.normalizer(image,
                                    target_size=self.target_size,
                                    interpolation=cv2.INTER_NEAREST)
            
            label = sample['label']
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)
            batch_image.append(image)
            batch_label.append(label)
            
            if self.debug_mode:
                debug_image.append(img_path)
                debug_label.append(label)
                
        batch_image = np.array(batch_image)
        batch_label, label_length = self.label_converter.encode(batch_label)

        if self.debug_mode:
            return batch_image, batch_label, label_length, debug_image, debug_label
        else:
            return batch_image, batch_label, label_length

    def on_epoch_end(self):
        if self.phase:
            self.dataset = shuffle(self.dataset)
