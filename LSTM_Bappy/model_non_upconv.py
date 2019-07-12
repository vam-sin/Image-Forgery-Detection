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
from keras.layers import LSTM, Reshape, Conv2DTranspose, Conv2D, MaxPooling2D, UpSampling2D, Dropout
import glob
import cv2
from keras.optimizers import SGD
import keras.backend as K

opt = SGD(lr=0.001, momentum=0.9)

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

train_datagen = ImageDataGenerator()

training_X = train_datagen.flow_from_directory('../COVERAGE/Forged_Im',
                                                 target_size = (374, 324),
                                                 batch_size = 1,
                                                 seed=1337,
                                                 class_mode = None, shuffle=False)

training_Y = train_datagen.flow_from_directory('../COVERAGE/Forged_Im_M',
                                            target_size = (8, 8),
                                            batch_size = 1,
                                            seed=1337,
                                            color_mode='grayscale',
                                            class_mode = None, shuffle=False)

dataset = zip(training_X, training_Y)

print('Dataset Prepared')

# Dataset Stats
# View X
print(training_X[0].shape)
res = np.squeeze(training_X[0], axis=0)
print(res.shape)
plt.imshow(res.astype('uint8'))
plt.show()
# # # View Y
res = np.squeeze(training_Y[0], axis=0)
res = np.squeeze(res, axis=-1)
print(res.shape)
plt.imshow(res.astype('uint8'))
plt.show()

# This shows that the images are not loaded in order. shuffle=False

# Stacked LSTM
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
    model.summary()

    return model

classifier = LSTM_Model()

checkpoint = ModelCheckpoint('model_lstm_non_upconv.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

classifier.fit_generator(dataset, steps_per_epoch=100, epochs=20, verbose=1, callbacks=callbacks_list)
