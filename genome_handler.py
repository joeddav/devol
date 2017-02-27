import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


class GenomeHandler:
    def __init__(self):
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
        self.convolutional_layer_shape = {
            # Present
            0: [0, 1],
            # Filters
            1: range(16, 257),
            # Batch Normalization
            2: [0, 1],
            # Activation
            3: self.activation.keys(),
            # Dropout
            4: [i / 20 for i in range(11)],
            # Max Pooling
            5: range(3),
        }

        self.dense_layer_shape = {
            # Present
            0: [0, 1],
            # Number of Nodes
            1: [2**i for i in range(5, 12)],
            # Batch Normalization
            2: [0, 1],
            # Activation
            3: self.activation.keys(),
            # Dropout
            4: [i / 20 for i in range(11)],
        }
        self.convolution_layers = 6
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.dense_layers = 3
        self.dense_layer_size = len(self.dense_layer_shape)

    def mutate(self, genome):
        while True:
            index = np.random.choice(range(1, len(genome)))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convolutional_layer_shape[range_index]
                    genome[index] = np.random.choice(choice_range)
                    break
            elif index != len(genome) - 1:
                offset = self.convolution_layer_size * self.convolution_layers
                new_index = (index - offset)
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.dense_layer_shape[range_index]
                    genome[index] = np.random.choice(choice_range)
                    break
            else:
                genome[index] = np.random.choice(self.optimizer.keys())
                break
        return genome

    def decode(self, genome):
        model = Sequential()
        offset = 0
        input_layer = True
        for i in range(self.convolution_layers):
            if genome[offset]:
                convolution = None
                if input_layer:
                    convolution = Convolution2D(
                                        genome[offset + 1], 3, 3,
                                        border_mode='same',
                                        input_shape=(1, 28, 28))
                    input_layer = False
                else:
                    convolution = Convolution2D(
                                        genome[offset + 1], 3, 3,
                                        border_mode='same')
                model.add(convolution)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(genome[offset + 4]))
                max_pooling_type = genome[offset + 5]
                if max_pooling_type == 1:
                    model.add(MaxPooling2D(pool_size=(2, 2), border_mode="same"))
                elif max_pooling_type == 2:
                    model.add(MaxPooling2D(pool_size=(3, 3), border_mode="same", strides=(2, 2)))
            offset += self.convolution_layer_size

        model.add(Flatten())

        for i in range(self.dense_layer_size):
            if genome[offset]:
                model.add(Dense(genome[offset + 1]))
            if genome[offset + 2]:
                model.add(BatchNormalization())
            model.add(Activation(self.activation[genome[offset + 3]]))
            model.add(Dropout(genome[offset + 4]))

        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer[genome[offset]],
                      metrics=['accuracy'])
        return model

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for j, r in self.convolutional_layer_shape.iteritems():
                genome.append(np.random.choice(r))
        for i in range(self.dense_layers):
            for j, r in self.dense_layer_shape.iteritems():
                genome.append(np.random.choice(r))
        genome.append(np.random.choice(self.optimizer.keys()))
        genome[0] = 1
        return genome
