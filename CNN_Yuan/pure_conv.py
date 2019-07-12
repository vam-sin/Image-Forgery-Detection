# Build a better loss function (Pixel wise Cross Entropy)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Reshape
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization as BN
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
import cv2
import glob
import numpy as np
from keras.optimizers import SGD
import matplotlib.pyplot as plt

def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)

    gen_dice_coef = 2*numerator/denominator

    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)




# opt = SGD(lr=0.01, momentum=0.9)
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
    # model.add(Reshape((81,69)))
    model.compile(loss=dice_loss,
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

classifier = Mask_Gen()

checkpoint = ModelCheckpoint('model_pure_cnn.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(shear_range=0.2,
                    zoom_range=0.2,
                    rotation_range=45,
                    horizontal_flip=True,
                    vertical_flip=True)

training_X = train_datagen.flow_from_directory('../COVERAGE/Forged_Im',
                                                 target_size = (374, 324),
                                                 batch_size = 16,
                                                 seed=1337,
                                                 class_mode = None, shuffle=False)

training_Y = train_datagen.flow_from_directory('../COVERAGE/Forged_Im_M',
                                            target_size = (81, 69),
                                            batch_size = 16,
                                            seed=1337,
                                            color_mode='grayscale',
                                            class_mode = None, shuffle=False)

# Dataset Stats
# View X
# print(training_Y[1].shape)
# # res = np.squeeze(training_X[1], axis=0)
# # print(res.shape)
# # plt.imshow(res.astype('uint8'))
# # plt.show()
# # # View Y
# res = np.squeeze(training_Y[1], axis=0)
# res = np.squeeze(res, axis=-1)
# print(res.shape)
# plt.imshow(res.astype('uint8'))
# plt.show()

dataset = zip(training_X, training_Y)

# print('Data loaded.')
# # Checking the dataset
classifier.fit_generator(dataset, steps_per_epoch=100,epochs=20, verbose=1, callbacks=callbacks_list)
classifier.save_weights("model_pure_cnn.h5")
