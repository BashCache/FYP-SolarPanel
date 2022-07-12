import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import opening, disk, closing
from pylab import *
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import scipy
import glob
from skimage.io import imread, imshow
from skimage import img_as_ubyte

def white_patch(image, percentile=100):
    white_patch_image = img_as_ubyte((image*1.0 / 
                                   np.percentile(image,percentile,
                                   axis=(0, 1))).clip(0, 1))
    return white_patch_image

def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged