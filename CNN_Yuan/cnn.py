# Image Dimensions
# from PIL import Image
#
# im = Image.open('/home/vamsi/Vamsi/IGIB/Datasets/SplicingDetection/Train/Pristine/canong3_02_sub_02.tif')
# print(im.size)
# Size of the images is (757, 568)

# Updates
# First Test: loss: 0.2991 - acc: 0.8921 - val_loss: 1.5942 - val_acc: 0.6318
# Possible Overfitting // Need more data
# Added Dropout and regularization.
# Changed Batch size to 16 from 32 and steps per eppoch to 100 from 50 to have enough memory.
# Second Test :

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
import cv2

def Yuan():
    model = Sequential()
    model.add(Conv2D(30,kernel_size = (5,5),
        strides=(1,1),padding='valid',activation='relu',input_shape=(757,568,3)))
    model.add(Conv2D(30,kernel_size=(5,5),
        strides=2,activation='relu'))
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
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid')

    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

classifier = Yuan()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('SplicingDetection/Train',
                                                 target_size = (757, 568),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('SplicingDetection/Test',
                                            target_size = (757, 568),
                                            batch_size = 32,
                                            class_mode = 'binary')

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

classifier.fit_generator(training_set,
                         steps_per_epoch = 50,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 50,
                         callbacks = callbacks_list)

# Prediction for the example
from PIL import Image
import numpy as np

img  = Image.open('example_p.tif')
print(img.size)

x = np.expand_dims(img, axis=0)
print(x.shape)

model.predict(x)
classes = train_datagen.class_indices
print(classes)
