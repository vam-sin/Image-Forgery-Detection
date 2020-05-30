# Work to do
# Change batch size and learning rate, tweak the model.
# The pipeline and mask generator are perfect.

# Libraries
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization as BN
from keras.backend import squeeze
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM, Reshape, Conv2DTranspose, Conv2D, MaxPooling2D, UpSampling2D
import glob
import cv2
from keras.optimizers import SGD

opt = SGD(lr=0.001, momentum=0.9)

train_datagen = ImageDataGenerator()

training_X = train_datagen.flow_from_directory('../COVERAGE/Forged_Im',
                                                 target_size = (374, 324),
                                                 batch_size = 1,
                                                 color_mode='rgb',
                                                 class_mode = None, shuffle=False,
                                                 subset='training')

training_Y = train_datagen.flow_from_directory('../COVERAGE/Forged_Im_M',
                                            target_size = (248, 248),
                                            batch_size = 1,
                                            color_mode='grayscale',
                                            class_mode = None, shuffle=False,
                                            subset='training')

dataset = zip(training_X, training_Y)

print('Dataset Prepared')

# # Understanding the data
# im = training_X[1][0]
# print(im.shape)
# # The images are loaded as an array
# plt.imshow(im.astype('uint8'))
# plt.show()
#
# y = training_Y[1][0]
# print(y.shape)
# y = np.squeeze(y, axis=-1)
# # The images are loaded as an array
# print(y.shape)
# plt.imshow(y.astype('uint8'))
# plt.show()

# This shows that the images are not loaded in order. shuffle=False

# Stacked LSTM
def LSTM_Model():
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(374, 324, 3)))
    model.add(BN())
    model.add(Conv2D(30, kernel_size=(5,5), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Reshape((1,672)))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(64*3, activation='tanh', return_sequences=True))
    model.add(Reshape((8,8,3)))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(1, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.compile(optimizer = opt, loss = 'mse', metrics=['accuracy'])
    model.summary()

    return model

# classifier = LSTM_Model()

checkpoint = ModelCheckpoint('model_lstm.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# classifier.fit_generator(dataset, steps_per_epoch=100, epochs=50, verbose=1, callbacks=callbacks_list)
