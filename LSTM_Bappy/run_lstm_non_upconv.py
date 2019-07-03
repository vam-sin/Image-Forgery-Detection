from PIL import Image
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import mixem
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization as BN
from keras.backend import squeeze
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM, Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, MaxPooling2D, UpSampling2D
import glob
import cv2
from keras.optimizers import SGD

opt = SGD(lr=0.1, momentum=0.9)

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
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(Reshape((8,8,1)))
    model.compile(optimizer = opt, loss = 'mse', metrics=['accuracy'])
    #model.summary()

    return model


classifier = LSTM_Model()
classifier.load_weights("model_lstm_non_upconv.h5")
print('Loaded Weights')

# Predictions
img  = Image.open('6t.tif')
img = cv2.resize(np.float32(img), dsize=(324, 374), interpolation=cv2.INTER_CUBIC)
print(img.shape)
x = np.expand_dims(img, axis=0)
print(x.shape)

x = np.expand_dims(img, axis=0)

res = classifier.predict(x)
print(res.shape)
res = np.squeeze(res, axis=0)
res = np.squeeze(res, axis=-1)
plt.imshow(res.astype('uint8'))
plt.show()
