import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class Rotate:
    def __init__(self, angle, padding_color=None, keep_shape=True):
        self.angle = angle
        self.padding_color = padding_color
        self.keep_shape = keep_shape
        
    def __call__(self, image):
        try:
            height, width, _ = image.shape
        except:
            height, width    = image.shape
            
        center_x, center_y = (width // 2, height // 2)
        
        M = cv2.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)
        
        if self.padding_color:
            if isinstance(self.padding_color, int):
                fill_color = [self.padding_color, self.padding_color, self.padding_color]
            else:
                fill_color = self.padding_color
        else:
            fill_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        if not self.keep_shape:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
    
            # compute the new bounding dimensions of the image
            nW = int((height * sin) + (width * cos))
            nH = int((height * cos) + (width * sin))
    
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - center_x
            M[1, 2] += (nH / 2) - center_y
    
            # perform the actual rotation and return the image
            img = cv2.warpAffine(image.copy(), M, (nW, nH), borderValue=fill_color)
        else:
            img = cv2.warpAffine(image.copy(), M, (width, height), borderValue=fill_color)
        return img


class RandomRotate:
    def __init__(self, angle_range=15, padding_color=None, keep_shape=True, prob=0.5):
        self.angle_range   = angle_range
        self.padding_color = padding_color
        self.keep_shape    = keep_shape
        self.prob          = prob

    def __call__(self, image):
        angle = np.random.randint(-self.angle_range, self.angle_range)
        self.aug        = Rotate(angle, padding_color=self.padding_color, keep_shape=self.keep_shape)
        
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            image = self.aug(image)
        return image