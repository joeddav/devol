import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from genome_handler import GenomeHandler
import random as rand

# Our genetic algorithm.
# This will contain methods for fitness, crossover, mutation, etc.
class Evolution:

    def __init__(self):
        self.genome_handler = GenomeHandler()
        self.loadMNIST()
        self.bssf = (0, None, 0) # fitness (1/loss), model, accuracy

    def loadMNIST(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
        self.x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)

    # Create a population and evolve
    def run(self, num_generations, pop_size):
        # Generate initial random population
        members = np.array([self.genome_handler.generate() for _ in range(pop_size)])
        fit = np.array([self.fitness(member) for member in members])
        print "Population #1"
        pop = Population(members, fit)

        # Evolve over generations
        for i in range(1, num_generations):
            members = []
            for i in range(int(pop_size*0.95)): # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            for i in range(int(pop_size*0.95), pop_size): # Carryover
                members.append(pop.select())
            for i in range(len(members)): # Mutation
                if rand.uniform(0, 1) < 0.01:
                    members[i] = self.mutate(members[i])
            member = np.array(member)
            fit = np.array([self.fitness(member) for member in members])
            print "Population #" + str(i + 1)
            pop = Population(members, fit)

        # persist the best model
        self.bssf[1].save("keras_model")

    # Returns the accuracy for a model as 1 / loss
    def fitness(self, genome):
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        try:
            model.fit(self.x_train, self.y_train, \
                    validation_data=(self.x_test, self.y_test),
                    nb_epoch=10, batch_size=200, verbose=0)
            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        except: # this is a temporary fix addressing models that train (b.c. too many max poolings, etc.)
            loss = 1
            accuracy = 0
        fitness = 1 / loss

        # keep the best fit model as we go
        if fitness > self.bssf[0]:
            self.bssf = (fitness, model, accuracy)

        return fitness
    
    def crossover(self, genome1, genome2):
        genome1 = genome1.tolist()
        genome2 = genome2.tolist()
        #swap the genomes split at the crossover index
        crossIndexA = rand.randint(0, len(genome1))
        genome1TempA = genome1[:crossIndexA] + genome2[crossIndexA:]
        genome2TempA = genome2[:crossIndexA] + genome1[crossIndexA:]

        #swap the genomes from the first split, split at the second crossover index
        crossIndexB = crossIndexA + rand.randint(0, len(genome1) - crossIndexA)
        genome1TempB = genome1TempA[:crossIndexB] + genome2TempA[crossIndexB:]
        genome2TempB = genome2TempA[:crossIndexB] + genome1TempA[crossIndexB:]

        return np.array([genome1TempB, genome2TempB][rand.randint(0, 1)])
    
    def mutate(self, genome):
        return self.genome_handler.mutate(genome)

class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses):
        self.members = members
        self.fitnesses = fitnesses
        self.s_fit = sum(self.fitnesses)
        self.printStats()

    def printStats(self):
        print "Best Fitness:", max(self.fitnesses)
        print "Average Fitness:", np.mean(self.fitnesses)
        print "Standard Deviation Fitness:", np.std(self.fitnesses)
    
    def select(self):
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.fitnesses[i]
            if sum_fits > dart:
                return self.members[i]
