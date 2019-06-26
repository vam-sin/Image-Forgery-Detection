# Libraries
from PIL import Image
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import mixem
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM, Dense, Flatten
import glob
import cv2

# Dimensions
# Output: (429976, 3)
# Input: (374, 324, 3)
# Output_Image: (385, 389, 3)
# Output Data

# Data Import
images_forged = [cv2.imread(file) for file in glob.glob("COVERAGE/Forged_Images/*.tif")]
images_pristine = [cv2.imread(file) for file in glob.glob("COVERAGE/Pristine_Images/*.tif")]

forged_output = []
pristine_output = []
for i in range(len(images_forged)):
    one = np.ones((1,1))
    forged_output.append(one)
    zero = np.zeros((1,1))
    pristine_output.append(zero)

# im = Image.open('example_p.tif')
# print(im.size)
# plt.imshow(im)
# plt.show()

# sobel_im = []
# # Linear Filter (Sobel)
# for im in images:
#     sx = ndimage.sobel(im, axis=0, mode='constant')
#     sy = ndimage.sobel(im, axis=1, mode='constant')
#     sob = np.hypot(sx, sy)
#     sobel_im.append(sob)

# sob_im = Image.fromarray(sob, 'RGB')
# sob_im.show()
# sob = sobel_im[0]
# # print(sob.shape)
# sarr = np.asarray(im)
# # print(sarr.shape)
# arr = sarr.reshape(sarr.shape[0]*sarr.shape[1], sarr.shape[2])
# # print(arr.shape)
# arr = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))

# Output Masks


# # Expectation Maximization
# weights, distributions, ll = mixem.em(arr, [
#     mixem.distribution.MultivariateNormalDistribution(np.array((2, 50)), np.identity(3)),
#     mixem.distribution.MultivariateNormalDistribution(np.array((4, 80)), np.identity(3)),
#     mixem.distribution.MultivariateNormalDistribution(np.array((6, 110)), np.identity(3)),
# ])

images_forged = np.asarray(images_forged)
print(images_forged.shape)
images_pristine = np.asarray(images_pristine)
print(images_pristine.shape)
forged_output = np.asarray(forged_output)
print(forged_output.shape)
pristine_output = np.asarray(pristine_output)
print(pristine_output.shape)

# Stacked LSTM
model = Sequential()
model.add(LSTM(64,activation='tanh',return_sequences=True, input_shape=(images_forged[0].shape[0]*images_forged[0].shape[1], images_forged[0].shape[2])))
model.add(LSTM(64,return_sequences=True, activation='tanh'))
model.add(LSTM(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model.summary()

# Training on Forged
for i in range(len(images_forged)):
    images_forged[i] = images_forged[i].reshape(images_forged[i].shape[0]*images_forged[i].shape[1], images_forged[i].shape[2])
    images_forged[i] = np.reshape(images_forged[i], (1, images_forged[i].shape[0], images_forged[i].shape[1]))
    model.fit(images_forged[i], forged_output[i], epochs=10, verbose=1)
    images_pristine[i] = images_pristine[i].reshape(images_pristine[i].shape[0]*images_pristine[i].shape[1], images_pristine[i].shape[2])
    images_pristine[i] = np.reshape(images_pristine[i], (1, images_pristine[i].shape[0], images_pristine[i].shape[1]))
    model.fit(images_pristine[i], pristine_output[i], epochs=10, verbose=1)
model.save('model.h5')
