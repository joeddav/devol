from __future__ import print_function

from genome_handler import GenomeHandler
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist, cifar10
from keras.callbacks import EarlyStopping
from datetime import datetime
import random as rand
import csv
from tqdm import trange, tqdm
import sys

class DEvol:

    def __init__(self, genome_handler, data_path=""):
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')
        self.bssf = (None, 0.) # model, accuracy
        print("Genome encoding and accuracy data stored at", self.datafile, "\n")
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            genome = genome_handler.genome_representation() + ["Val Loss", "Val Accuracy"]
            writer.writerow(genome)

    # Create a population and evolve
    # Returns best model found in the form of (model, accuracy)
    def run(self, dataset, num_generations, pop_size, epochs, fitness=None):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        # Generate initial random population
        members = [self.genome_handler.generate() for _ in range(pop_size)]
        fit = []
        for i in range(len(members)):
            print("\nmodel {0}/{1} - generation {2}/{3}:\n".format(i + 1, len(members), 1, num_generations))
            loss, acc, model = self.evaluate(members[i], epochs)
            if acc > self.bssf[1]:
                self.bssf = (model, acc)
            fit.append(acc)
        pop = Population(members, fit, fitness)
        fit = np.array(fit)
        print("Generation 1:\t\tmax: {0}\t\taverage: {1}\t\tstd: {2}".format(max(fit), np.mean(fit), np.std(fit)))

        # Evolve over generations
        for gen in range(1, num_generations):
            members = []
            for i in range(int(pop_size*0.95)): # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            members += pop.getBest(pop_size - int(pop_size*0.95))
            for i in range(len(members)): # Mutation
                members[i] = self.mutate(members[i], gen)
            fit = []
            for i in range(len(members)):
                print("\nmodel {0}/{1} - generation {2}/{3}:\n".format(i + 1, len(members), gen + 1, num_generations))
                loss, acc, model = self.evaluate(members[i], epochs)
                if acc > self.bssf[1]:
                    self.bssf = (model, acc)
                fit.append(acc)
            pop = Population(members, fit, fitness)
            fit = np.array(fit)
            print("Generation {3}:\t\tmax: {0}\t\taverage: {1}\t\tstd: {2}".format(max(fit), np.mean(fit), np.std(fit), gen + 1))
        
        return self.bssf

    def evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
            epochs=epochs,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=1)])
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)

        # Record the stats
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss, accuracy]
            writer.writerow(row)  

        return loss, accuracy, model
    
    def crossover(self, genome1, genome2):
        crossIndexA = rand.randint(0, len(genome1))
        child = genome1[:crossIndexA] + genome2[crossIndexA:]
        return child
    
    def mutate(self, genome, generation):
        num_mutations = max(3, generation / 4) # increase mutations as program continues
        return self.genome_handler.mutate(genome, num_mutations)

class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score):
        self.members = members
        fitnesses -= min(fitnesses)
        fitnesses /= max(fitnesses)
        self.scores = list(map(score or self.score, fitnesses))
        self.s_fit = sum(self.scores)

    def score(self, fitness):
        return (fitness * 100)**4

    def getBest(self, n):
        combined = [(self.members[i], self.scores[i]) \
                for i in range(len(self.members))]
        sorted(combined, key=(lambda x: x[1]), reverse=True)
        return [x[0] for x in combined[:n]]

    def select(self):
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits > dart:
                return self.members[i]
