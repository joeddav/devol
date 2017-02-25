import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Activation

# Just handles decoding from genome strings into Keras models
class GenomeDecoder:

    # Returns a compiled keras model
    def decode(self, genome):
        pass


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