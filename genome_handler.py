import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

class GenomeHandler:
    def __init__(self, max_conv_layers, max_dense_layers, max_filters, max_dense_nodes,
                input_shape, batch_normalization=True, dropout=True, max_pooling=True):
        self.optimizer = [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = [
            'relu',
            'sigmoid',
        ]
        self.convolutional_layer_shape = [
            # Present
            [0, 1],
            # Filters
            [2**i for i in range(3, int(math.log(max_filters, 2)) + 1)],
            # Batch Normalization
            [0, (1 if batch_normalization else 0)],
            # Activation
            range(len(self.activation)),
            # Dropout
            [(i if dropout else 0) for i in range(11)],
            # Max Pooling
            range(3) if max_pooling else 0,
        ]
        self.dense_layer_shape = [
            # Present
            [0, 1],
            # Number of Nodes
            [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            # Batch Normalization
            [0, (1 if batch_normalization else 0)],
            # Activation
            range(len(self.activation)),
            # Dropout
            [(i if dropout else 0) for i in range(11)],
        ]
        self.data_augmentation_values = [
            #featurewise_center
            [0, 1],
            #samplewise_center
            [0, 1],
            #featurewise_std_normalization
            [0, 1],
            #samplewise_std_normalization
            [0, 1],
            #zca_whitening
            [0, 1],
            #rotation_range
            [-10, 0, 10],
            #width_shift_range
            [0, 1, 2], # divide by 20
            #height_shift_range
            [0, 1, 2], # divide by 20
            #shear_range
            [0, 1, 2], # divide by 20
            #zoom_range
            [0, 1, 2, 3, 4], # divide by 20
            #channel_shift_range: this is for RGB images
            #cval: this is the constant fill and I think we want it to be 0
            #horizontal_flip
            [0, 1],
            #vertical_flip
            [0, 1],
        ]
        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.dense_layers = max_dense_layers - 1 # this doesn't include the softmax layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(range(1, len(genome)))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convolutional_layer_shape[range_index]
                    genome[index] = np.random.choice(choice_range)
		elif rand.uniform(0, 1) <= 0.01: # randomly flip deactivated layers
		    genome[index - index % self.convolution_layer_size] = 1
            elif index != len(genome) - 1:
                offset = self.convolution_layer_size * self.convolution_layers
                new_index = (index - offset)
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.dense_layer_shape[range_index]
                    genome[index] = np.random.choice(choice_range)
		elif rand.uniform(0, 1) <= 0.01:
		    genome[present_index + offset] = 1
            else:
                genome[index] = np.random.choice(self.optimizer.keys()) 
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
                                        genome[offset + 1], (3, 3),
                                        padding='same',
                                        input_shape=self.input_shape)
                    input_layer = False
                else:
                    convolution = Convolution2D(
                                        genome[offset + 1], (3, 3),
                                        padding='same')
                model.add(convolution)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
                max_pooling_type = genome[offset + 5]
                if max_pooling_type == 1:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
            offset += self.convolution_layer_size

        model.add(Flatten())

        for i in range(self.dense_layers):
            if genome[offset]:
                model.add(Dense(genome[offset + 1]))
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
            offset += self.dense_layer_size
        
        datagen = ImageDataGenerator(
            featurewise_center=(genome[offset] == 1),
            samplewise_center=(genome[offset + 1] == 1),
            featurewise_std_normalization=(genome[offset + 2] == 1),
            #samplewise_std_normalization=(genome[offset + 3] == 1),
            zca_whitening=(genome[offset + 4] == 1),
            rotation_range=int(genome[offset + 5]),
            width_shift_range=(float(genome[offset + 6]) / 20.0),
            height_shift_range=(float(genome[offset + 7]) / 20.0),
            shear_range=(float(genome[offset + 8]) / 20.0),
            zoom_range=(float(genome[offset + 9]) / 20.0),
            #channel_shift_range=(float(genome[offset + 10]) / 50),
            #cval=(float(genome[offset + 11]) / 50),
            horizontal_flip=(genome[offset + 10] == 1),
            vertical_flip=(genome[offset + 11] == 1),
            data_format="channels_first")

        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
		      optimizer=self.optimizer[genome[offset]],
		      metrics=["accuracy"])
        return model, datagen

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for r in self.convolutional_layer_shape:
                genome.append(np.random.choice(r))
        for i in range(self.dense_layers):
            for r in self.dense_layer_shape:
                genome.append(np.random.choice(r))
        genome.append(np.random.choice(range(len(self.optimizer))))
        for r in self.data_augmentation_values:
            genome.append(np.random.choice(r))
        genome[0] = 1
        return genome
