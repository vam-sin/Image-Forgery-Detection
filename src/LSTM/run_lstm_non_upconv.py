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
import keras.backend as K
from keras.layers import LSTM, Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, MaxPooling2D, Dropout
import glob
import cv2
from keras.optimizers import SGD

opt = SGD(lr=0.1, momentum=0.9)

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def LSTM_Model():
    model = Sequential()
    model.add(Conv2D(30,kernel_size = (5,5),
        strides=(1,1),padding='valid',activation='relu',input_shape=(374, 324, 3)))
    model.add(BN())
    model.add(Conv2D(30,kernel_size=(5,5),
        strides=1,activation='relu'))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(BN())
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(BN())
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(BN())
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(Dropout(0.5))
    model.add(BN())
    model.add(Conv2D(1,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(Reshape((1,5589)))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    # model.add(LSTM(64, activation='tanh', return_sequences=True))
    # model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(Reshape((8,8,1)))
    model.compile(optimizer = 'adam', loss = dice_loss, metrics=['accuracy'])
    # model.summary()

    return model


classifier = LSTM_Model()
classifier.load_weights("model_lstm_non_upconv_onelayer.h5")
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
