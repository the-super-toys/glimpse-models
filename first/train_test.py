import glob

from keras.layers import MaxPooling2D, Convolution2D, UpSampling2D, Activation, BatchNormalization, Flatten, Reshape
from keras.optimizers import Adam, Nadam
from scipy import misc
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt

from first.gaussian_kernel import gausian_kernel

image_list = []
heatmap_list = []

samples = 1
batch_size = 1
epochs = 200

for filename in glob.glob('../heatmaps/*.jpg')[3:3 + samples]:
    image_list.append(filename.replace("heatmaps", "images"))
    heatmap_list.append(filename)

print(len(image_list))
print(len(heatmap_list))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()  # only difference


image_cache = dict()
heatmap_cache = dict()


def _generator(batch_size=100):
    index = 0
    while 1:
        start = index
        end = index + batch_size
        index += batch_size
        if index >= samples:
            index = 0

        images = []
        heatmaps = []

        for file_path in image_list[start:end]:
            if file_path in image_cache:
                images.append(image_cache[file_path])
            else:
                loaded_image = misc.imread(file_path)
                image_cache[file_path] = loaded_image
                images.append(loaded_image)

        for file_path in heatmap_list[start:end]:
            if file_path in heatmap_cache:
                heatmaps.append(heatmap_cache[file_path])
            else:
                loaded_heatmap = misc.imread(file_path) / 255
                heatmap_cache[file_path] = loaded_heatmap
                heatmaps.append(loaded_heatmap)

        yield np.asarray(images) / 255, np.asarray(heatmaps).reshape(-1, 480, 640, 1)


generator = _generator(batch_size)

# model definition
model = Sequential()

model.add(MaxPooling2D((4, 4), input_shape=(480, 640, 3)))

model.add(Convolution2D(16, (3, 3), activation="relu", padding='same'))

# model.add(Convolution2D(8, (3, 3), activation="relu", padding='same'))
# model.add(Convolution2D(32, (3, 3), activation="relu", padding='same'))

# model.add(Convolution2D(32, (1, 1), activation="relu", padding='same', input_shape=(480, 640, 3)))
# model.add(Convolution2D(32, (1, 1), activation="relu", padding='same', input_shape=(480, 640, 3)))

model.add(Convolution2D(1, (3, 3), activation="relu", padding='same', input_shape=(480, 640, 3)))

kernel = gausian_kernel(5)
model.add(Convolution2D(1, (16, 16), kernel_initializer=kernel, padding='same', trainable=False))

model.add(UpSampling2D((4, 4)))
model.summary()

optimizer = Nadam()
# checkpoint = ModelCheckpoint('model.h5', verbose=1, monitor='loss', save_best_only=True, mode='auto')

model.compile(optimizer, 'mean_squared_logarithmic_error', ['mae'])  # mean_squared_logarithmic_error

history = model.fit_generator(generator, steps_per_epoch=samples / batch_size, epochs=epochs)

# Test image
for i in range(5):
    test_number = samples - 1 - i  # np.random.randint(0, samples)
    test_image = misc.imread(image_list[test_number])
    test_heatmap = misc.imread(heatmap_list[test_number])

    prediction = model.predict(np.asarray([test_image]))[0].reshape(480, 640)

    print(prediction.shape)
    print(prediction.max())
    print(prediction.min())

    plt.figure()
    plt.imshow(test_image)
    plt.show()

    plt.figure()
    plt.imshow(softmax(np.asarray(test_heatmap) / 255) * 255, cmap="Greys_r")
    plt.show()

    plt.figure()
    plt.imshow(prediction * 255, cmap="Greys_r")
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

# x, y = next(generator)

# print(x.shape)
# print(y.shape)
