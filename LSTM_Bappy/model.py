# Libraries
from PIL import Image
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import mixem
from keras.models import Sequential
from keras.layers import LSTM, Dense


im = Image.open('example_p.tif')
print(im.size)
# plt.imshow(im)
# plt.show()

# Linear Filter (Sobel)
sx = ndimage.sobel(im, axis=0, mode='constant')
sy = ndimage.sobel(im, axis=1, mode='constant')
sob = np.hypot(sx, sy)

print(sob.shape)
arr = np.asarray(sob)
print(arr.shape)
arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
print(arr.shape)
arr = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))

# # Expectation Maximization
# weights, distributions, ll = mixem.em(arr, [
#     mixem.distribution.MultivariateNormalDistribution(np.array((2, 50)), np.identity(3)),
#     mixem.distribution.MultivariateNormalDistribution(np.array((4, 80)), np.identity(3)),
#     mixem.distribution.MultivariateNormalDistribution(np.array((6, 110)), np.identity(3)),
# ])

# Stacked LSTM
model = Sequential()
model.add(LSTM(64,activation='tanh',return_sequences=True, input_shape=(1,3)))
model.add(LSTM(64,return_sequences=True, activation='tanh'))
model.add(LSTM(64, activation='tanh'))
model.add(Dense(2, activation = 'softmax')) # Resamples or not
model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])

model.summary()



print(model.predict(arr))
