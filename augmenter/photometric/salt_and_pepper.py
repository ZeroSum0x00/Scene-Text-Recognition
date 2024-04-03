import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


# class SaltAndPepper:
#     def __init__(self, ratio=0.5, phi=0.1):
#         assert 0 <= ratio <= 1.0, "ratio must be between 0.0 and 1.0"
#         self.ratio = ratio
#         self.phi = phi
        
#     def __call__(self, image):
#         img = image.copy()
#         try:
#             height, width, channels = img.shape
#         except:
#             height, width = img.shape
#             channels = 1
            
#         # Salt mode
#         num_salt = int(self.phi * height * width * self.ratio)
#         row_coords = np.random.randint(0, height, size=num_salt)
#         col_coords = np.random.randint(0, width, size=num_salt)
#         img[row_coords, col_coords, :] = [255, 255, channels]

#         # Pepper mode
#         num_pepper = int(self.phi * height * width * (1.0 - self.ratio))
#         row_coords = np.random.randint(0, height, size=num_pepper)
#         col_coords = np.random.randint(0, width, size=num_pepper)
#         img[row_coords, col_coords, :] = [0, 0, channels]
#         return img


# class RandomSaltAndPepper:
#     def __init__(self, ratio_range=0.5, phi_range=0.1, prob=0.5):
#         self.ratio_range = ratio_range
#         self.phi_range   = phi_range
#         self.prob        = prob

#     def __call__(self, image):
#         if isinstance(self.ratio_range, (list, tuple)):
#             ratio = float(np.random.choice(self.ratio_range))
#         else:
#             ratio = float(np.random.uniform(0, self.ratio_range))
            
#         if isinstance(self.phi_range, (list, tuple)):
#             phi = float(np.random.choice(self.phi_range))
#         else:
#             phi = float(np.random.uniform(0, self.phi_range))
            
#         self.aug        = SaltAndPepper(ratio, phi)
        
#         p = np.random.uniform(0, 1)
#         if p >= (1.0-self.prob):
#             image = self.aug(image)
#         return image


class SaltAndPepper:
    def __init__(self, phi=0.1):
        assert 0 <= phi <= 1.0, "phi must be between 0.0 and 1.0"
        self.phi = phi
        
    def __call__(self, image):
        try:
            img = image.copy()
            dtype = img.dtype
            intensity_levels = 2 ** (img[0, 0].nbytes * 8)
            min_intensity, max_intensity = 0, intensity_levels - 1
            random_image = np.random.choice([min_intensity, 1, np.nan], p=[self.phi / 2, 1 - self.phi, self.phi / 2], size=img.shape)
            img = img.astype(np.float32) * random_image
            img = np.nan_to_num(img, nan=max_intensity).astype(dtype)
            return img
        except:
            return image


class RandomSaltAndPepper:
    def __init__(self, phi_range=0.05, prob=0.5):
        self.phi_range   = phi_range
        self.prob        = prob

    def __call__(self, image):
        if isinstance(self.phi_range, (list, tuple)):
            phi = float(np.random.choice(self.phi_range))
        else:
            phi = float(np.random.uniform(0, self.phi_range))

        aug = SaltAndPepper(phi)
        
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            image = aug(image)
        return image
