import re
import cv2
import random
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

from data_utils.data_processing import extract_data_folder, get_data, Normalizer
from models.layers.label_converter import CTCLabelConverter
from visualizer.visual_value import tensor_value_info

def get_train_test_data(data_zipfile,
                        annotation_dir,
                        dst_dir,
                        target_size,
                        batch_size,
                        character,
                        max_string_length,
                        sensitive,
                        augmentor=None,
                        normalizer='divide',
                        data_type='txt',
                        check_data=False,
                        load_memory=False):
    data_folder = extract_data_folder(data_zipfile, dst_dir)

    data_train = get_data(data_folder, 
                          annotation_dir    = annotation_dir,
                          character         = character,
                          data_type         = data_type,
                          max_string_length = max_string_length,
                          sensitive         = sensitive,
                          phase             = 'train',
                          check_data        = check_data,
                          load_memory       = load_memory)
    train_generator = Train_Data_Sequence(data_train, 
                                          target_size=target_size, 
                                          batch_size=batch_size, 
                                          character=character,
                                          max_string_length=max_string_length,
                                          augmentor=augmentor,
                                          normalizer=normalizer)

    data_valid = get_data(data_folder, 
                          annotation_dir    = annotation_dir,
                          character         = character,
                          data_type         = data_type,
                          max_string_length = max_string_length,
                          sensitive         = sensitive,
                          phase             = 'validation',
                          check_data        = check_data,
                          load_memory       = load_memory)
    valid_generator = Valid_Data_Sequence(data_valid, 
                                          target_size=target_size, 
                                          batch_size=batch_size, 
                                          character=character,
                                          max_string_length=max_string_length,
                                          normalizer=normalizer)
    
    # data_test, _ = get_data(data_folder, phase='test')
    # test_generator = Test_Data_Sequence(data_test, 
    #                                     target_size=target_size, 
    #                                     batch_size=batch_size, 
    #                                     classes=class_names,
    #                                     normalizer=normalizer)

    return train_generator, valid_generator


class Train_Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size, 
                 batch_size, 
                 character, 
                 max_string_length=25, 
                 augmentor=None, 
                 normalizer=None):
        self.data_path = dataset['data_path']
        self.dataset = dataset['data_extractor']
        self.batch_size = batch_size
        self.character = character
        self.target_size = (batch_size, *target_size)

        self.dataset = shuffle(self.dataset)

        self.N = self.n = len(self.dataset)

        self.normalizer = Normalizer(normalizer)
        
        self.max_string_length = max_string_length
        self.label_converter = CTCLabelConverter(character)
        
    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image    = []
        batch_label    = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.N
            sample = self.dataset[i]
            img_path = self.data_path + sample['filename']
            image = cv2.imread(img_path)
            image = self.normalizer(image,
                                    target_size=self.target_size[1:],
                                    interpolation=cv2.INTER_NEAREST)
            
            label = sample['label']
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label.lower())
            batch_image.append(image)
            batch_label.append(label)

        batch_image = np.array(batch_image)
        batch_label, label_length = self.label_converter.encode(batch_label, max_string_length=self.max_string_length)
        return batch_image, batch_label, label_length


class Valid_Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size, 
                 batch_size, 
                 character, 
                 max_string_length=25, 
                 augmentor=None, 
                 normalizer=None):
        self.data_path = dataset['data_path']
        self.dataset = dataset['data_extractor']
        self.batch_size = batch_size
        self.character = character
        self.target_size = (batch_size, *target_size)

        self.N = self.n = len(self.dataset)

        self.normalizer = Normalizer(normalizer)
        
        self.max_string_length = max_string_length
        self.label_converter = CTCLabelConverter(character)
        
    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image    = []
        batch_label    = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.N
            sample = self.dataset[i]
            img_path = self.data_path + sample['filename']
            images = cv2.imread(img_path)
            orin_img = images
            images = self.normalizer(images,
                                target_size=self.target_size[1:],
                                interpolation=cv2.INTER_NEAREST)
            
            label = sample['label']
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label.lower())

            batch_image.append(images)
            batch_label.append(label)

        batch_image = np.array(batch_image)
        batch_label, label_length = self.label_converter.encode(batch_label, max_string_length=self.max_string_length)
        return batch_image, batch_label, label_length
    

# class Test_Data_Sequence(Sequence):
#     def __init__(self, dataset, target_size, character, batch_size, augmentor=None, normalizer=None):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.target_size = (batch_size, target_size[0], target_size[1], target_size[2])
#         self.character = character

#         from sklearn.utils import shuffle
#         self.dataset = shuffle(self.dataset, random_state=0)

#         self.N = self.n = len(self.dataset)

#         self.normalizer = Normalizer(normalizer)

#     def __len__(self):
#         return int(np.ceil(self.N / float(self.batch_size)))

#     def __getitem__(self, idx):
#         batch_x = np.zeros(self.target_size, dtype=np.float32)
#         batch_y = []

#         for i in range(self.batch_size):          
#             index = min((idx * self.batch_size) + i, self.N)
#             if index < self.N:
#                 sample = self.dataset[index]
#             else:
#                 sample = random.choice(self.dataset)

#             x = cv2.imread(sample[0], 1)

#             x = self.normalizer(x,
#                                 mean=0.2,
#                                 target_size=self.target_size[1:],
#                                 interpolation=cv2.INTER_NEAREST)
            
#             y = sample[1]
#             out_of_char = f'[^{self.character}]'
#             y = re.sub(out_of_char, '', y.lower())

#             batch_x[i] = x
#             batch_y.append(y)

#         return batch_x, batch_y