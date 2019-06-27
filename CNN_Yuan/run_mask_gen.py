from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization as BN
import keras
import cv2
import numpy as np
from keras.optimizers import SGD

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
classifier.load_weights("model_1.h5")
print('Loaded Weights')

# Prediction for the example

img  = Image.open('6t.tif')
img = cv2.resize(np.float32(img), dsize=(324, 374), interpolation=cv2.INTER_CUBIC)
print(img.shape)
x = np.expand_dims(img, axis=0)
print(x.shape)

x = np.expand_dims(img, axis=0)

res = classifier.predict(x)
res = np.squeeze(res, axis=0)
print(res.shape)
res = np.array(res,np.int32)
# plt.imshow((res * 255).astype(np.uint8))
res = Image.fromarray(res, 'RGB')
res.save('output.png')
res.show()
