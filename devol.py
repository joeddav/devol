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
import sys
import operator
import gc

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


class DEvol:

    def __init__(self, genome_handler, data_path=""):
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')

        print("Genome encoding and accuracy data stored at", self.datafile, "\n")
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            genome = genome_handler.genome_representation() + ["Val Loss", "Val Accuracy"]
            writer.writerow(genome)

    def set_objective(self, metric):
        """set the metric and objective for this search  should be 'accuracy' or 'loss'"""
        if metric is 'acc':
            metric = 'accuracy'
        if not metric in ['loss', 'accuracy']:
            raise ValueError(
                'Invalid metric name {} provided - should be "accuracy" or "loss"'.format(metric))
        self.metric = metric
        self.objective = "max" if self.metric is "accuracy" else "min"
        self.metric_index = 1 if self.metric is 'loss' else -1
        self.metric_op = METRIC_OPS[self.objective is 'max']
        self.metric_objective = METRIC_OBJECTIVES[self.objective is 'max']


    def run(self, dataset, num_generations, pop_size, epochs, fitness=None, metric='accuracy'):
        """run genetic search on dataset given number of generations and population size

        Args:
            dataset : tuple or list of numpy arrays in form ((train_data, train_labels), (validation_data, validation_labels))
            num_generations (int): number of generations to search
            pop_size (int): initial population size
            epochs (int): epochs to run each search, passed to keras model.fit -currently searches are
                            curtailed if no improvement is seen in 1 epoch
            fitness (None, optional): scoring function to be applied to population scores, will be called on a numpy array
                                      which is a  min/max scaled version of evaluated model metrics, so
                                      It should accept a real number including 0. If left as default just the min/max
                                      scaled values will be used.
            metric (str, optional): must be "accuracy" or "loss" , defines what to optimize during search

        Returns:
            (keras model, float, float ): best model found in the form of (model, loss, accuracy)
        """
        self.set_objective(metric)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        # Generate initial random population
        members = [self.genome_handler.generate() for _ in range(pop_size)]
        fit = []
        metric_index = 1 if self.metric is 'loss' else -1
        for i in range(len(members)):
            print("\nmodel {0}/{1} - generation {2}/{3}:\n"\
                    .format(i + 1, len(members), 1, num_generations))
            res = self.evaluate(members[i], epochs)
            v = res[metric_index]
            del res
            fit.append(v)

        fit = np.array(fit)
        pop = Population(members, fit, fitness, obj=self.objective)
        print("Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}"\
                .format(self.metric_objective(fit), np.mean(fit), np.std(fit), 1, self.metric))

        # Evolve over 
        for gen in range(1, num_generations):
            members = []
            for i in range(int(pop_size * 0.95)):  # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            members += pop.getBest(pop_size - int(pop_size * 0.95))
            for i in range(len(members)):  # Mutation
                members[i] = self.mutate(members[i], gen)
            fit = []
            for i in range(len(members)):
                print("\nmodel {0}/{1} - generation {2}/{3}:\n"
                        .format(i + 1, len(members), gen + 1, num_generations))
                res = self.evaluate(members[i], epochs)
                v = res[metric_index]
                del res
                fit.append(v)

            fit = np.array(fit)
            pop = Population(members, fit, fitness, obj=self.objective)
            print("Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}"\
                    .format(self.metric_objective(fit), np.mean(fit), np.std(fit), gen + 1, self.metric))

        return self.genome_handler.decode_best(self.datafile)

    def evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        try:
            model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                      epochs=epochs,
                      verbose=1,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=1)])
            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        except:
            loss = 1.5
            accuracy = 1 / self.genome_handler.n_classes
            gc.collect()
            print("An error occurred and the model could not train. Assigned poor score.")
        # Record the stats
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss, accuracy]
            writer.writerow(row)
        return model, loss, accuracy

    def crossover(self, genome1, genome2):
        crossIndexA = rand.randint(0, len(genome1))
        child = genome1[:crossIndexA] + genome2[crossIndexA:]
        return child

    def mutate(self, genome, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(genome, num_mutations)


class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        scores = fitnesses - fitnesses.min()
        if scores.max() > 0:
            scores /= scores.max()
        if obj is 'min':
            scores = 1 - scores
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)

    def getBest(self, n):
        combined = [(self.members[i], self.scores[i])
                    for i in range(len(self.members))]
        sorted(combined, key=(lambda x: x[1]), reverse=True)
        return [x[0] for x in combined[:n]]

    def select(self):
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]
