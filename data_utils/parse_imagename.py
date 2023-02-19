import os
import cv2
from tqdm import tqdm


class ParseImageName:
    def __init__(self, 
                 data_dir,
                 character,
                 max_string_length,
                 sensitive,
                 load_memory,
                 check_data):
        self.data_dir          = data_dir
        self.character         = character
        self.max_string_length = max_string_length
        self.sensitive         = sensitive
        self.load_memory       = load_memory
        self.check_data        = check_data

    def __call__(self, image_files):
        data_extraction = []
        for filename in tqdm(image_files, desc="Load dataset"):
            info_dict = {}
            info_dict['filename'] = filename
            info_dict['label'] = text = ''.join(filename.split('.')[0].split('_')[1:])
            info_dict['lenght'] = len(text)
            if len(text) == 0:                
                del info_dict['filename']
                del info_dict['label']
                del info_dict['lenght']
            if self.max_string_length and (len(text) > self.max_string_length):
                del info_dict['filename']
                del info_dict['label']
                del info_dict['lenght']
            try:
                for t in info_dict['label']:
                    if t not in self.character:
                        del info_dict['filename']
                        del info_dict['label']
                        del info_dict['lenght']
                        break
            except:
                pass

            if info_dict:
                data_extraction.append(info_dict)
        return data_extraction
