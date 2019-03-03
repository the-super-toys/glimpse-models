# PyTorch Implementation

This folder contains 3 files:

[net.ipynb](https://github.com/the-super-toys/glimpse-models/blob/master/pytorch/net.ipynb): Define and train the model. We use a very small custom architecture based on
ResNet and the model is trained on 20000 annotated images and validated on 5000.

[inference.ipynb](https://github.com/the-super-toys/glimpse-models/blob/master/pytorch/inference.ipynb): Once the model is trained, this file allows you to test the model on specific
images and evaluate the latency on your computer's CPU. Latency on a computer is not equivalent to latency on mobile devices.
Based on our experience, on mobile devices the latency is at least x3 slower than on a laptop, meanin that a model with
a 30ms latency on a computer has at least 90ms latency on a mobile device.

The best achieved latency on a computer is 9ms but in the device it goes to up to 70ms :(

[torch2tf.ipynb](https://github.com/the-super-toys/glimpse-models/blob/master/pytorch/torch2tf.ipynb): In this last notebook, we convert the PyTorch model to a tensorflow lite
model that can be used in an Android device.