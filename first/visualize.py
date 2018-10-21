import numpy as np

import matplotlib.pyplot as plt
from PythonAPI.salicon.salicon import SALICON
import skimage.io as io


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()  # only difference


# initialize salicon dataset
salicon = SALICON("../annotations/fixations_train2014examples.json")

print("getting image IDs")
imgIds = salicon.getImgIds()
img = salicon.loadImgs(imgIds[0])[0]

print("plotting image "+img['file_name'])
image = io.imread('../images/' + img['file_name'])
plt.figure()
plt.imshow(image)
plt.show()

print("plotting annotations")
annIds = salicon.getAnnIds(imgIds=img['id'])
anns = salicon.loadAnns(annIds)
salicon.showAnns(anns)
plt.show()

print("plotting softmax heatmap")
heatmap = softmax(salicon.buildFixMap(anns))
plt.figure()
plt.imshow(heatmap, cmap="Greys_r")
plt.show()

print(heatmap.max(), heatmap.min())
