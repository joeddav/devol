import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation
from keras.datasets import mnist
from genome_handler import GenomeHandler
import random as rand

# Our genetic algorithm.
# This will contain methods for fitness, crossover, mutation, etc.
class Evolutions:

    def __init__(self):
        self.genome_handler = GenomeHandler()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    # Run the whole genetic algorithm
    def run(self, num_generations, pop_size):
        # Generate initial random population
        members = np.array([self.genome_handler.generate() for _ in range(pop_size)])
        fit = np.array([self.fitness(member) for member in members])
        pop = Population(members, fit)
        # for _ in range(num_generations):
        #     # crossover
        #     for i in range()

        # 1) generate population
        # 2) for num_generations create new populations:
            # 1) select two parents for crossover
            # 2) select one parent for mutation
            # generate fitness for each

    # Returns the accuracy for a model
    def fitness(self, genome):
        model = self.genome_handler.decode(genome)
        model.fit(self.x_train, self.y_train, \
                validation_data=(self.x_test, self.y_test),
                nb_epoch=10, batch_size=50, verbose=0)
        scores = model.evaluate(X_test, y_test, verbose=0)
        return scores[0]
    
    # Crossover two genomes
    def crossover(self, genome1, genome2):
        pass

    # Mutate one gene
    def mutate(self, genome):
        pass

class Population:

    def __init__(self, members, fitnesses):
        self.members = members
        self.fitnesses = fitnesses
        self.s_fit = sum(self.fitnesses)
    
    def select(self):
        dart = rand.randint(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.fitnesses[i]
            if sum_fits > dart:
                return self.members[i]


    
