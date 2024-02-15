import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class GaussianBlur:
    def __init__(self, kernel=3, sigma=5):
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        self.sigma  = sigma
        
    def __call__(self, image):
        img = image.copy()
        img = cv2.GaussianBlur(img, self.kernel, 7.3)
        return img


class RandomGaussianBlur:
    def __init__(self, kernel=3, sigma_range=5, prob=0.5):
        self.kernel      = kernel
        self.sigma_range = sigma_range
        self.prob        = prob
        
    def __call__(self, image):
        if isinstance(self.sigma_range, (list, tuple)):
            sigma = float(np.random.choice(self.sigma_range))
        else:
            sigma = float(np.random.uniform(0, self.sigma_range))
            
        self.aug = GaussianBlur(self.kernel, sigma)
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            image = self.aug(image)
        return image