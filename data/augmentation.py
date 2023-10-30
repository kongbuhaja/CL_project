import numpy as np
import cv2

class Copy():
    def __init__(self):
        pass
    
    def __call__(self, image):
        return image

class Padding():
    def __init__(self):
        pass

    def __call__(self, image):
        h, w, c = image.shape
        l = max(h, w)
        new_image = np.zeros((l, l, c)).astype(np.uint8)
        
        y, x = (l - h)//2, (l - w)//2
        new_image[y:y+h, x:x+w] = image
        return new_image

class Resize():
    def __init__(self, new_size):
        self.new_size = new_size * 2 if len(new_size)==1 else new_size

    def __call__(self, image):
        return cv2.resize(image, self.new_size)

class Normalization():
    def __init__(self):
        pass

    def __call__(self, image):
        return image/255.