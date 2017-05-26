from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from devol import DEvol
from genome_handler import GenomeHandler
import numpy as np
from sklearn import datasets
from random import shuffle

def prepData(x, y, train_prop=0.75):
    combined = [(x[i], y[i]) for i in range(len(x))]
    shuffle(combined)
    x = np.array([item[0] for item in combined])
    y = np.array([item[1] for item in combined])
    ind = int(train_prop * len(combined))
    return ((x[:ind], y[:ind]), (x[ind:], y[ind:]))

iris = datasets.load_iris()
x = iris.data
y = to_categorical(iris.target)

# define model constraints

max_conv_layers = 0
max_dense_layers = 3 # including final softmax layer
max_conv_kernals = 0
max_dense_nodes = 1024
input_shape = x.shape[1:]
num_classes = 3

genome_handler = GenomeHandler(max_conv_layers, max_dense_layers, max_conv_kernals, \
                    max_dense_nodes, input_shape, num_classes)

# create and run a DEvol

num_generations = 5
population_size = 5
num_epochs = 20

devol = DEvol(genome_handler)
devol.run(prepData(x, y), num_generations, population_size, num_epochs)