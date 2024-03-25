import os
import cv2
from tqdm import tqdm
from utils.files import valid_image


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
            info_dict['image'] = None
            text = ''.join(filename.split('.')[0].split('_')[1:])
            info_dict['label'] = text if self.sensitive else text.lower()
            info_dict['lenght'] = len(text)
            info_dict['path'] = self.data_dir
            image_path = os.path.join(self.data_dir, filename)

            if self.check_data:
                try:
                    valid_image(image_path)
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    shape = img.shape
                except Exception as e:
                    os.remove(image_path)
                    print(f"Error: File {filename} is can't loaded: {e}")
                    continue
                    
            if self.load_memory:
                img = cv2.imread(image_path)
                info_dict['image'] = img
                
            if len(text) == 0:                
                del info_dict['filename']
                del info_dict['image']
                del info_dict['label']
                del info_dict['lenght']
                del info_dict['path']

            if self.max_string_length and (len(text) > self.max_string_length):
                del info_dict['filename']
                del info_dict['image']
                del info_dict['label']
                del info_dict['lenght']
                del info_dict['path']
                
            try:
                for t in info_dict['label']:
                    if t not in self.character:
                        del info_dict['filename']
                        del info_dict['image']
                        del info_dict['label']
                        del info_dict['lenght']
                        del info_dict['path']
                        break
            except:
                pass

            if info_dict:
                data_extraction.append(info_dict)
        return data_extraction
