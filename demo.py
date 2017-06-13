from __future__ import print_function

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from devol import DEvol, GenomeHandler
import numpy as np
from keras import backend as K

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used
# for many introductory deep learning examples. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

K.set_image_data_format("channels_last")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

max_conv_layers = 6
max_dense_layers = 2 # including final softmax layer
max_conv_kernals = 256
max_dense_nodes = 1024
input_shape = x_train.shape[1:]
num_classes = 10

genome_handler = GenomeHandler(max_conv_layers, max_dense_layers, max_conv_kernals, \
                    max_dense_nodes, input_shape, num_classes)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.

num_generations = 10
population_size = 10
num_epochs = 1

devol = DEvol(genome_handler)
model, loss, accuracy = devol.run(dataset, num_generations, population_size, num_epochs)
model.summary()
