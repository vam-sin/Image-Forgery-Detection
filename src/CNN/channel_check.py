from PIL import Image
import cv2


image = cv2.imread('dataset/original/one_orig.jpeg')
print(image.shape)
if image.ndim == 2:

    channels = 1 #single (grayscale)

if image.ndim == 3:

    channels = image.shape[-1]

print(channels)
