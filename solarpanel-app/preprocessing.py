import cv2
import numpy as np

def resize_image(img):    
    X = []
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
    X.append(resized_img)
    X = np.array(X)
    X = X.astype('float32')
    X /= 255.
    return X