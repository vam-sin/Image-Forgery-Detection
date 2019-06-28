# Libraries
from PIL import Image
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import mixem
from keras.models import Sequential
from keras.backend import squeeze
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM, Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, MaxPooling2D, TimeDistributed
import glob
import cv2
from keras.optimizers import SGD

opt = SGD(lr=0.005, momentum=0.9)

# Dimensions
# Output: (429976, 3)
# Input: (374, 324, 3)
# Output_Image: (385, 389, 3)
# Output Data

train_datagen = ImageDataGenerator()

training_X = train_datagen.flow_from_directory('../COVERAGE/Forged_Im',
                                                 target_size = (374, 324),
                                                 batch_size = 1,
                                                 class_mode = None)

training_Y = train_datagen.flow_from_directory('../COVERAGE/Forged_Im_M',
                                            target_size = (8, 8),
                                            batch_size = 1,
                                            class_mode = None)

dataset = zip(training_X, training_Y)

print('Dataset Prepared')
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

# Stacked LSTM
def LSTM_Model():
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(374, 324, 3)))
    model.add(Conv2D(30, kernel_size=(5,5), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Reshape((1,672)))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(64*3, activation='tanh', return_sequences=True))
    model.add(Reshape((8,8,3)))
    model.compile(optimizer = opt, loss = 'mse', metrics=['accuracy'])
    model.summary()

    return model

classifier = LSTM_Model()

checkpoint = ModelCheckpoint('model_lstm.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

classifier.fit_generator(dataset, steps_per_epoch=100, epochs=20, verbose=1, callbacks=callbacks_list)

# # Training on Forged
# for i in range(len(images_forged)):
#     images_forged[i] = images_forged[i].reshape(images_forged[i].shape[0]*images_forged[i].shape[1], images_forged[i].shape[2])
#     images_forged[i] = np.reshape(images_forged[i], (1, images_forged[i].shape[0], images_forged[i].shape[1]))
#     model.fit(images_forged[i], forged_output[i], epochs=10, verbose=1)
#     images_pristine[i] = images_pristine[i].reshape(images_pristine[i].shape[0]*images_pristine[i].shape[1], images_pristine[i].shape[2])
#     images_pristine[i] = np.reshape(images_pristine[i], (1, images_pristine[i].shape[0], images_pristine[i].shape[1]))
#     model.fit(images_pristine[i], pristine_output[i], epochs=10, verbose=1)
# model.save('model.h5')
