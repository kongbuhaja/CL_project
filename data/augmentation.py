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

class Random_resize():
    def __init__(self, minimum=0.7, maximum=1.3):
        self.min = minimum
        self.max = maximum

    def __call__(self, image):
        new_size = np.tile(np.random.uniform(self.min, self.max , 1), [2,])
        org_size = image.shape[0:2]
        return cv2.resize(cv2.resize(image, new_size), org_size)

class Rotate90():
    def __init__(self):
        pass

    def __call__(self, image):
        if np.random.uniform(0,1) < 0.25:
            return image

        else:
            n = np.random.uniform(1,4).astype(np.int32)
            return np.rot90(image, n)

class Random_Vflip():
    def __init__(self):
        pass
    
    def __call__(self, image):
        if np.random.unifrom(0,1) < 0.5:
            return image
        
        else:
            return image[:,::-1]


class Normalization():
    def __init__(self):
        pass

    def __call__(self, image):
        return image/255.