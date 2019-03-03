# Keras implementation

This folder contains 2 files:

[net.ipynb](https://github.com/the-super-toys/glimpse-models/blob/master/keras/net.ipynb): Define and train the model. We use a very small custom architecture based on
ResNet and the model is trained on 20000 annotated images and validated on 5000.

[keras_to_tf.ipynb](https://github.com/the-super-toys/glimpse-models/blob/master/keras/keras_to_tl.ipynb): In this notebook, we convert the Keras model to a tensorflow lite
model which can be used on an Android device.

Keras implementation is not currently used on client libraries because it has more latency than PyTorch implenebtation when running it on devices. It goes to up to 100ms :(
