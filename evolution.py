import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation

# Our genetic algorithm.
# This will contain methods for fitness, crossover, mutation, etc.
class Evolutions:

    # Run the whole thing
    def run(self):
        pass

    # Evaluate the fitness of a model
    def fitness(self, model):
        pass

    # Crossover two genomes
    def crossover(self, genome1, genome2):
        pass

    # Mutate one gene
    def mutate(self, genome):
        pass
