import os
import cv2
import json
from tqdm import tqdm


class ParseJSON:
    def __init__(self, 
                 data_dir,
                 annotation_file,
                 character,
                 max_string_length,
                 sensitive,
                 load_memory,
                 check_data):
        self.data_dir          = data_dir
        self.annotation_path   = annotation_file
        json_file = open(annotation_file)
        self.raw_data = json.load(json_file)
        json_file.close()

        self.character         = character
        self.max_string_length = max_string_length
        self.sensitive         = sensitive
        self.load_memory       = load_memory
        self.check_data        = check_data

    def __call__(self):
        data_extraction = []
        for filename, text in tqdm(self.raw_data.items(), desc="Load dataset"):
            info_dict = {}
            info_dict['filename'] = filename
            info_dict['label'] = text if self.sensitive else text.lower()
            info_dict['lenght'] = len(text)
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