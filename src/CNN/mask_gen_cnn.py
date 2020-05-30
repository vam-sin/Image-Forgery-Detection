# Model Stats
# Mask Size: (364, 316, 3)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
import cv2
import glob
import numpy as np

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

def Mask_Gen():
    model = Sequential()
    model.add(Conv2D(30,kernel_size = (5,5),
        strides=(1,1),padding='valid',activation='relu',input_shape=(374, 324, 3)))
    model.add(Conv2D(30,kernel_size=(5,5),
        strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
    model.add(Conv2D(16,kernel_size=(3,3),
        strides=1,activation='relu'))
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

classifier.fit_generator(dataset, steps_per_epoch=100,epochs=20 verbose=1)
classifier.save_weights("model_masks.h5")

# # Prediction for the example
from PIL import Image
import numpy as np

classifier.load_weights("model_masks.h5")

img  = Image.open('example_p.tif')
img = cv2.resize(cv2.UMAT(img), dsize=(324, 374), interpolation=cv2.INTER_CUBIC)

x = np.expand_dims(img, axis=0)

res = classifier.predict(x)
