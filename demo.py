from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from devol import DEvol
from genome_handler import GenomeHandler
import numpy as np

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used 
# for many introductory deep learning examples. Here, we load the data and 
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon 
# models in a particular genetic program. In this example, a genome is 
# allowed **up to** 6 convolutional layeres, 3 dense layers, 256 feature 
# maps in each convolution, and 1024 nodes in each dense layer. It also 
# specifies three possible activation functions. See `genome-handler.py` 
# for more information.

genome_handler = GenomeHandler(6, 3, 256, 1024, x_train.shape[1:], 
                               activations=["relu", "sigmoid"])

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify 
# a few settings pertaining to the genetic program. In this example, we 
# have 10 generations of evolution, 20 members in each population, and 1 
# epoch of training used to evaluate each model's fitness. The program 
# will save each genome's encoding, as well as the model's loss and 
# accuracy, in a `.csv` file printed at the beginning of program.

devol = DEvol(genome_handler)
devol.run(dataset, 10, 20, 1)
