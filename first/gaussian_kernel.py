import cv2

import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, SeparableConv2D, Conv2D
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
import matplotlib.pyplot as plt


def gausian_kernel(sigma):
    def gausian_kernel_gen(shape, dtype=None):
        kernel = np.zeros(shape, dtype=dtype)
        kernel[shape[0] // 2, shape[1] // 2] = 1
        return gaussian_filter(kernel, sigma=sigma)

    return gausian_kernel_gen

#
# kernel = gausian_kernel(8)
#
# plt.figure()
# plt.imshow(kernel((31, 31)) * 255, cmap="Greys_r")
# plt.show()
#
# model = Sequential()
# model.add(Conv2D(1, (31, 31), kernel_initializer=kernel, input_shape=(480, 640, 1),
#                           padding='same',
#                           trainable=False))
# model.summary()
# model.compile("adam", 'mse')
#
# test_image = misc.imread("../images/COCO_train2014_000000000110.jpg")
# test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY).reshape(480, 640, 1)
# blurred_image = model.predict(np.asarray([test_image]))[0]
#
# print(test_image.shape)
# print(blurred_image.shape)
#
# plt.figure()
# plt.imshow(test_image.reshape(480, 640), cmap="Greys_r")
# plt.show()
#
# plt.figure()
# plt.imshow(blurred_image.reshape(480, 640), cmap="Greys_r")
# plt.show()
