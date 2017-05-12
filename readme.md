# DEvol - Deep Neural Network Evolution

DEvol (DeepEvolution) utilizes genetic programming to automatically architect a deep neural network with optimal hyperparameters for a given dataset using the Keras library. This approach should design an equal or superior model to what a human could design when working under the same constraints as are imposed upon the genetic program (e.g. number of layers allowed, etc.). The current setup is designed for classification problems, though this could be easily extended.

See `mnist example.ipynb` for a simple example.

## Evolution

Each model is represented as fixed-width genome encoding information about the network's structure. In the current very simple setup, a model contains a number of convolutional layers, a number of dense layers, and an optimzer. The convolutional layers can be evolved to include varying numbers of feature maps, activation functions, dropout, batch normalization, and max pooling. Each dense layer can do the same, minus the max pooling. The complexity of these models could easily be extended beyond what we have here to include any parameters included in Keras.

Below is a highly simplified visualization of how genetic crossover might take place between two models.

<img width="75%" src="https://preview.ibb.co/gdMDak/crossover.png">
<i>Genetic crossover and mutation of neural networks</i>

## Results

For demonstration, we ran our program on the MNIST dataset (see `mnist-example.ipynb` for an example setup) with 20 generations and a population size of 50. We allowed the model up to 6 convolutional layers and 4 dense layers (including the softmax layer). The best accuracy we attained with 10 epochs of training under these constraints was 99.4%, which is higher than we were able to achieve under the same constraints when designing on our own. Below is a visualization of our running max.

Keep in mind that these results are obtained with simple, relatively shallow neural networks with no data augmentation, transfer learning, ensembling, fine-tuning, or any other fancy optimization approach. Virtually any of these techniques could be incorporated into the genetic program, however. 

<img width="75%" src="https://preview.ibb.co/i4BDak/running_max.png">
<i>Running max of MNIST accuracies across 20 generations</i>

## Application

The biggest barrier with this approach is time. With neural nets taking as long as they do to train as is, optimization through training of hundreds or neural networks is not always feasible. Below are some approaches to combat this issue:

- **Early Stopping** - You don't need to train a model for 10 epochs if it's not going anywhere. Cut it off early.
- **Train for Less Epochs** - Training in a genetic program serves one purpose: to evaulate a model's fitness in relation to the other models. It may not be necessary to train for 100 epochs just to obtain this comparison. You may only need 2 or 3. The only thing to watch out for with this is evolving models that converge early. 
- **Hyper-Hyper-Parameter Selection** - The more robust you allow your models to be, the longer it will take to converge. I.e. don't allow horizontal flipping on a character recognition problem just because the genetic program will eventually learn this itself. The less space the program has to explore, the faster it will arrive. 

For some problems, it may be ideal to just plug the data into a genetic program and let the program build you an optimal model. For some this is infeasable. In either case, just running the genetic program for a while could give you insights you may have never considered for optimal model design. In running our program on MNIST, we learned that ReLU does far better in convolutional layers, but that Sigmoid does just fine on dense layers as well. We learned that ADAGRAD was the best performing pre-built optimizer. We gained insights on the number of nodes to include in each dense layer. 

At worst, DEvol could help you improve your own model design through observation. At best, it will give you a beautiful, finely-tuned model architecture. 

## Wanna Try It?

DEvol is pretty straightforward to use for basic classification problems. See `mnist-example.ipynb` for an example. There are three basic steps:

1. **Prep your dataset.** DEvol expects your labels to be one-hot encoded as it uses `categorical_crossentropy` for its loss function. Otherwise, you can prep your data however you'd like. Just pass your input shape into `GenomeHandler`.
2. **Create a `GenomeHandler`.** The `GenomeHandler` defines the constraints that you apply to your models. Specify the max number of convolutional and dense layers, the max dense nodes and feature maps, and the input shape. You can also specify whether you'd like to allow batch_normalization, dropout, and max_pooling, which are by default true. You can also pass in a list of optimizers and activation functions you'd like to allow.
3. **Create and run the DEvol.** Just pass in your `GenomeHandler` to the `DEvol` constructor and then run it. Here you have a few more options such as the number of generations, the population size, epochs used for fitness evaluation, and an (optional) fitness function which converts a model's accuracy into a fitness score.

See `mnist-example.ipynb` for a basic example.
