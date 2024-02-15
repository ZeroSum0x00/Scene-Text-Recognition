import os
import sys
import cv2
import lmdb
import numpy as np
from tqdm import tqdm


class ParseLMDB:
    def __init__(self, 
                 annotation_file,
                 character,
                 max_string_length,
                 sensitive,
                 check_data):
        self.lmdb_env = lmdb.open(annotation_file, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.lmdb_env:
            print('cannot create lmdb from %s' % (annotation_file))
            sys.exit(0)
            
        self.character         = character
        self.max_string_length = max_string_length
        self.sensitive         = sensitive
        self.check_data        = check_data

    def __call__(self):
        data_extraction = []
        with self.lmdb_env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            for index in tqdm(range(nSamples)):
                info_dict = {}
                info_dict['filename'] = None
                index += 1  # lmdb starts with 1
                img_key = 'image-%09d'.encode() % index
                imgbuf = txn.get(img_key)
                imgbuf = np.frombuffer(imgbuf, dtype=np.uint8)
                info_dict['image'] = cv2.imdecode(imgbuf, cv2.IMREAD_COLOR)
                label_key = 'label-%09d'.encode() % index
                text = txn.get(label_key).decode('utf-8')
                info_dict['label'] = text if self.sensitive else text.lower()
                info_dict['lenght'] = len(text)

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