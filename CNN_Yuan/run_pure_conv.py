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
classifier.load_weights("model_pure_cnn_dice.h5")
print('Loaded Weights')

# Prediction for the example
img  = Image.open('dataset/original/three.jpeg')
img = cv2.resize(np.float32(img), dsize=(324, 374), interpolation=cv2.INTER_CUBIC)
# If only one channel
img = cv2.merge((img,img,img))
print(img.shape)
x = np.expand_dims(img, axis=0)
print(x.shape)

x = np.expand_dims(img, axis=0)

res = classifier.predict(x)
print(res.shape)
res = np.squeeze(res, axis=0)
res = np.squeeze(res, axis=-1)
print(res.shape)
labels = {1:'Forged',2:'Pristine'}
plt.imshow(res.astype('uint8'))
cbar = plt.colorbar()
cbar.ax.set_yticklabels(['Pristine','','','','','Forged'])
cbar.set_label('Forgery Index')
plt.show()

# the area of the image with the largest intensity value
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
print(maxLoc)
x_max = maxLoc[0] + 10
x_min = maxLoc[0] - 10
y_max = maxLoc[1] + 10
y_min = maxLoc[1] - 10
# print(x_max)
object = cv2.imread('dataset/original/three.jpeg')
# cv2.imshow('Orig', object)
object = cv2.resize(object, dsize=(81, 69), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('Resize', object)
cv2.rectangle(object,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3)
# object = cv2.resize(np.float32(object), dsize=(81, 69), interpolation=cv2.INTER_CUBIC)

cv2.imshow('Box',object)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('Results/Dice/three_bound_box_dice.png',object)
