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
        self.new_size = np.array(new_size * 2 if len(new_size)==1 else new_size).astype(np.int32)

    def __call__(self, image):
        h, w, c = image.shape
        l = max(h, w)
        ratio = np.array([h/l, w/l])
        out_size = (ratio * self.new_size).astype(np.int32)
        return cv2.resize(image, out_size).reshape(*out_size[::-1], c)

class Random_resize():
    def __init__(self, new_size, minimum=0.7, maximum=1.3):
        self.new_size = np.array(new_size * 2 if len(new_size)==1 else new_size).astype(np.int32)
        self.min = minimum
        self.max = maximum

    def __call__(self, image):
        h, w, c = image.shape
        l = max(h, w)
        ratio = np.array([h/l, w/l])
        out_size = (ratio * self.new_size).astype(np.int32)
        down_size = (ratio * self.new_size * np.tile(np.random.uniform(self.min, self.max , 1), [2,])).astype(np.int32)
        return cv2.resize(cv2.resize(image, down_size), out_size).reshape(*out_size[::-1], c)

class Rotate90():
    def __init__(self):
        pass

    def __call__(self, image):
        if np.random.uniform(0,1) < 0.25:
            return image

        else:
            n = int(np.random.uniform(1,4))
            return np.rot90(image, n, axes = (0, 1))

class Random_Vflip():
    def __init__(self):
        pass
    
    def __call__(self, image):
        if np.random.uniform(0,1) < 0.5:
            return image
        else:
            return image[:,::-1]

class Normalization():
    def __init__(self):
        pass

    def __call__(self, image):
        return image/255.
    
class BGR2RGB():
    def __init__(self):
        pass

    def __call__(self, image):
        return image[..., [2, 1, 0]]