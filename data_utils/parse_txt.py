import os
import cv2
from tqdm import tqdm


class ParseTXT:
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
        txt_file = open(annotation_file, "r")
        self.raw_data = txt_file.readlines()
        txt_file.close()

        self.character         = character
        self.max_string_length = max_string_length
        self.sensitive         = sensitive
        self.load_memory       = load_memory
        self.check_data        = check_data

    def __call__(self):
        data_extraction = []
        for line in tqdm(self.raw_data, desc="Load dataset"):
            info_dict = {}
            filename, text = line.strip().split('\t')
            info_dict['filename'] = filename
            info_dict['image'] = None
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
                
            if self.max_string_length and (len(text) > self.max_string_length):
                del info_dict['filename']
                del info_dict['image']
                del info_dict['label']
                del info_dict['lenght']
                
            try:
                for t in info_dict['label']:
                    if t not in self.character:
                        del info_dict['filename']
                        del info_dict['image']
                        del info_dict['label']
                        del info_dict['lenght']
                        break
            except:
                pass

            if info_dict:
                data_extraction.append(info_dict)
        return data_extraction