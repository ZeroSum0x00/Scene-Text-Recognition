import os
import cv2
import lmdb
import numpy as np
from tqdm import tqdm
import sys

def createDataset(lmdb_path, save_path, mode='overwrite'):
    lmdb_env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not lmdb_env:
        print('cannot create lmdb from %s' % (lmdb_path))
        sys.exit(0)
    
    os.makedirs(save_path, exist_ok=True)

    with lmdb_env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        for index in tqdm(range(nSamples)):
            index += 1  # lmdb starts with 1
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            imgbuf = np.frombuffer(imgbuf, dtype=np.uint8)
            image = cv2.imdecode(imgbuf, cv2.IMREAD_COLOR)
            label_key = 'label-%09d'.encode() % index
            text = txn.get(label_key).decode('utf-8')
            new_image_path = os.path.join(save_path, f'{index}_{text}.jpg')

            if mode == 'overwrite':
                cv2.imwrite(new_image_path, image)
            else:
                if not os.path.isfile(new_image_path):
                    cv2.imwrite(new_image_path, image)

if __name__ == "__main__":
    lmdb_path = '/home/vbpo-101386/Desktop/mouting/dri_data/TuNIT/Datasets/OCR/lmdb_dataset/data_lmdb_release/data_lmdb_release/data_lmdb_release/training/ST'
    save_path = '/home/vbpo-101386/Desktop/TuNIT/Datasets/Text Recognition/SynthText_filename/train'
    
    createDataset(lmdb_path, save_path, mode='')