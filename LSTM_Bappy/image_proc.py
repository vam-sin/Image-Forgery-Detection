from PIL import Image
from scipy import misc
from scipy import ndimage
import cv2
import numpy as np

im = Image.open('1forged.tif')
im = np.asarray(im)
print(im.shape)

im2 = Image.open('6t.tif')
im2 = np.asarray(im2)
print(im2.shape)

# Masks have only dimension, the input images have three channels.
