import numpy as np


class GenomeHandler:
    def __init__(self):
        self.convolution_layers = 6
        self.dense_layers = 3
        self.optimizer = {
            0: 'adam',
            1: 'rmsprop',
            2: 'adagrad',
            3: 'delta',
            4: 'nag',
        }
        self.activation = {
            0: 'relu',
            1: 'sigmoid',
        }

    def decode(self, genome):
        pass

    def generate(self):
        genome = np.empty(0)
        # Optimizer
        genome = np.append(genome, np.random.choice(self.optimizer.keys()))
        for i in range(self.convolution_layers):
            # Present
            genome = np.append(genome, np.random.choice([0, 1]))
            # Filters
            genome = np.append(genome, np.random.choice(range(16, 257)))
            # Batch Normalization
            genome = np.append(genome, np.random.choice([0, 1]))
            # Activation
            genome = np.append(
                        genome, np.random.choice(self.activation.keys()))
            # Dropout
            genome = np.append(genome, np.random.uniform(.1, .6))
            # Max Pooling
            genome = np.append(genome, np.random.choice(range(3)))
        for i in range(self.dense_layers):
            # Present
            genome = np.append(genome, np.random.choice([0, 1]))
            # Number of Nodes
            genome = np.append(
                        genome, np.random.choice([2**i for i in range(5, 12)]))
            # Batch Normalization
            genome = np.append(genome, np.random.choice([0, 1]))
            # Activation
            genome = np.append(
                        genome, np.random.choice(self.activation.keys()))
            # Dropout
            genome = np.append(genome, np.random.uniform(.1, .6))
