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
classifier.load_weights("model_pure_cnn.h5")
print('Loaded Weights')

# Prediction for the example
img  = Image.open('6t.tif')
img = cv2.resize(np.float32(img), dsize=(324, 374), interpolation=cv2.INTER_CUBIC)
# If only one channel
# img = cv2.merge((img,img,img))
print(img.shape)
x = np.expand_dims(img, axis=0)
print(x.shape)

x = np.expand_dims(img, axis=0)

res = classifier.predict(x)
print(res.shape)
res = np.squeeze(res, axis=0)
res = np.squeeze(res, axis=-1)
print(res.shape)
plt.imshow(res.astype('uint8'))
plt.show()
