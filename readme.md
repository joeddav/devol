# DEvol - Deep Neural Network Evolution

DEvol (DeepEvolution) utilizes genetic programming to automatically architect a deep neural network with optimal hyperparameters for a given dataset using the Keras library. This approach should design an equal or superior model to what a human could design when working under the same constraints as are imposed upon the genetic program (e.g., maximum number of layers, maximum number of convolutional filters per layer, etc.). The current setup is designed for classification problems, though this could be extended to include any other output type as well.

See `demo.ipynb` for a simple example.

## Evolution

Each model is represented as fixed-width genome encoding information about the network's structure. In the current setup, a model contains a number of convolutional layers, a number of dense layers, and an optimizer. The convolutional layers can be evolved to include varying numbers of feature maps, different activation functions, varying proportions of dropout, and whether to perform batch normalization and/or max pooling. The same options are available for the dense layers with the exception of max pooling. The complexity of these models could easily be extended beyond these capabilities to include any parameters included in Keras, allowing the creation of more complex architectures.

Below is a highly simplified visualization of how genetic crossover might take place between two models.

<img width="75%" src="https://preview.ibb.co/gdMDak/crossover.png">
<i>Genetic crossover and mutation of neural networks</i>

## Results

For demonstration, we ran our program on the MNIST dataset (see `demo.ipynb` for an example setup) with 20 generations and a population size of 50. We allowed the model up to 6 convolutional layers and 4 dense layers (including the softmax layer). The best accuracy we attained with 10 epochs of training under these constraints was 99.4%, which is higher than we were able to achieve when manually constructing our own models under the same constraints. The graphic below displays the running maximum accuracy for all 1000 nets as they evolve over 20 generations.

Keep in mind that these results are obtained with simple, relatively shallow neural networks with no data augmentation, transfer learning, ensembling, fine-tuning, or other optimization techniques. However, virtually any of these methods could be incorporated into the genetic program. 

<img width="75%" src="https://preview.ibb.co/i4BDak/running_max.png">
<i>Running max of MNIST accuracies across 20 generations</i>

## Application

The most significant barrier in using DEvol on a real problem is the complexity of the algorithm. Because training neural networks is often such a computationally expensive process, training hundreds or thousands of different models to evaluate the fitness of each is not always feasible. Below are some approaches to combat this issue:

- **Parallel Training** - The nature of evaluating the fitness of multiple members of a population simultaneously is *embarassingly parallel*. A task like this would be trivial to distribute among many GPUs and even machines.
- **Early Stopping** - There's no need to train a model for 10 epochs if it stops improving after 3; cut it off early.
- **Train on Fewer Epochs** - Training in a genetic program serves one purpose: to evaluate a model's fitness in relation to other models. It may not be necessary to train to convergence to make this comparison; you may only need 2 or 3 epochs. However, it is important you exercise caution in decreasing training time because doing so could create evolutionary pressure toward simpler models that converge quickly. This creates a trade-off between training time and accuracy which, depending on the application, may or may not be desirable. 
- **Parameter Selection** - The more robust you allow your models to be, the longer it will take to converge; i.e., don't allow horizontal flipping on a character recognition problem even though the genetic program will eventually learn not to include it. The less space the program has to explore, the faster it will arrive at an optimal solution. 

For some problems, it may be ideal to simply plug the data into DEvol and let the program build a complete model for you, but for others, this hands-off approach may not be feasible. In either case, DEvol could give you insights into optimal model design that you may not have considered on your own. For the MNIST digit classification problem, we found that ReLU does far better than a sigmoid function in convolutional layers, but they work about equally well in dense layers. We also found that ADAGRAD was the highest-performing prebuilt optimizer and gained insight on the number of nodes to include in each dense layer. 

At worst, DEvol could give you insight into improving your model architecture. At best, it could give you a beautiful, finely-tuned model. 

## Wanna Try It?

To setup, just clone the repo and run `pip install -e path/to/repo`. You should then be able to access `devol` globally.

DEvol is pretty straightforward to use for basic classification problems. See `demo.ipynb` for an example. There are three basic steps:

1. **Prep your dataset.** DEvol expects a classification problem with labels that are one-hot encoded as it uses `categorical_crossentropy` for its loss function. Otherwise, you can prep your data however you'd like. Just pass your input shape into `GenomeHandler`.
2. **Create a `GenomeHandler`.** The `GenomeHandler` defines the constraints that you apply to your models. Specify the maximum number of convolutional and dense layers, the max dense nodes and feature maps, and the input shape. You can also specify whether you'd like to allow batch_normalization, dropout, and max_pooling, which are included by default. You can also pass in a list of optimizers and activation functions you'd like to allow.
3. **Create and run the DEvol.** Pass your `GenomeHandler` to the `DEvol` constructor, and run it. Here you have a few more options such as the number of generations, the population size, epochs used for fitness evaluation, the evaluation metric to optimize (accuracy or loss) and an (optional) fitness function which converts a model's accuracy or loss into a fitness score.

See `demo.ipynb` for a basic example.
