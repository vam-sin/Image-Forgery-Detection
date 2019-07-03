
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization as BN
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
import cv2
import glob
import numpy as np
from keras.optimizers import SGD

opt = SGD(lr=1, momentum=0.9)
def Mask_Gen():
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
    model.compile(loss='mse',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

classifier = Mask_Gen()

checkpoint = ModelCheckpoint('model_cnn.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    rotation_range=45,
                    horizontal_flip=True,
                    vertical_flip=True)

training_X = train_datagen.flow_from_directory('../COVERAGE/Forged_Im',
                                                 target_size = (374, 324),
                                                 batch_size = 8,
                                                 class_mode = None, shuffle=False)

training_Y = train_datagen.flow_from_directory('../COVERAGE/Forged_Im_M',
                                            target_size = (81, 69),
                                            batch_size = 8,
                                            color_mode='grayscale',
                                            class_mode = None, shuffle=False)
dataset = zip(training_X, training_Y)
print('Data loaded.')

classifier.fit_generator(dataset, steps_per_epoch=100,epochs=20, verbose=1, callbacks=callbacks_list)
classifier.save_weights("model_cnn.h5")
