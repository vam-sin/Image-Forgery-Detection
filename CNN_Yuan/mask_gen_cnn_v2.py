# Model Stats
# Mask Size: (364, 316, 3)

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

# Import Images
# Forged Images
# forged_images = [cv2.imread(file) for file in glob.glob("../COVERAGE/Forged_Images/*.tif")]
# # forged_images = np.asarray(forged_images)
# # Masks
# masks = [cv2.imread(file) for file in glob.glob("../COVERAGE/Forged_Images_Masks/*.tif")]
# masks_reshaped = []
# for m in masks:
#     m_res = cv2.resize(m, dsize=(316, 364), interpolation=cv2.INTER_CUBIC)
#     masks_reshaped.append(m_res)
# # masks_reshaped = np.asarray(masks_reshaped)
# print(forged_images[0].shape)
# print(masks_reshaped[0].shape)

# print('Data loaded.')
opt = SGD(lr=0.005, momentum=0.9)
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
    model.add(BN())
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2DTranspose(16, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(Conv2DTranspose(3, kernel_size=(3,3),
        strides=1, activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.compile(loss='mse',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

classifier = Mask_Gen()

# classifier.fit(forged_images, masks_reshaped, epochs=1, verbose=1)
# model.save('model.h5')
checkpoint = ModelCheckpoint('model_1.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator()

training_X = train_datagen.flow_from_directory('../COVERAGE/Forged_Im',
                                                 target_size = (374, 324),
                                                 batch_size = 4,
                                                 class_mode = None)

training_Y = train_datagen.flow_from_directory('../COVERAGE/Forged_Im_M',
                                            target_size = (364, 316),
                                            batch_size = 4,
                                            class_mode = None)
dataset = zip(training_X, training_Y)

classifier.fit_generator(dataset, steps_per_epoch=100,epochs=100, verbose=1, callbacks=callbacks_list)
classifier.save_weights("model_masks.h5")
