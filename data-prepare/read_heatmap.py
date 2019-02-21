import glob

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()  # only difference


image_list = []
images = []

print("reading file names...")
for filename in glob.glob('../heatmaps/*.jpg')[:100]:
    images.append(misc.imread(filename))
    image_list.append(filename)

print(len(image_list))
print(image_list[0])

heatmap = misc.imread(image_list[0]) / 255
heatmap = softmax(np.asarray(heatmap))

print(heatmap.shape)
print(heatmap.max())

print("plotting softmax heatmap")
plt.figure()
plt.imshow(heatmap, cmap="Greys_r")
plt.show()
