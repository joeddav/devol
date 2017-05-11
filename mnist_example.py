from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from devol import Devol
from genome_handler import GenomeHandler
import numpy as np

# prep dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))

# specifiy genome restrictions
genome_handler = GenomeHandler(6, 3, 256, 1024, x_train.shape[1:])

# create and run genetic program
devol = Devol(genome_handler)
devol.run(dataset, 10, 2, 1)