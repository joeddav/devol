# Contributing to DEvol

DEvol was created as a simple proof-of-concept project that was designed to allow a user to experiment easily with their own CNN classification problem. With the amount of interest that it has gathered, it's clear that it should be expanded into a more robust library that data scientists and AI specialists can experiment with reliably to gather insights about their model architectures. I have limited time I can devote to this project, so involvement from from the community will be necessary to make this happen. If you would like to contribute, please ensure that you are familiar with the project's objectives and then look at the `Projects` page.

## Vision

- **Extend for different types of nets.** Currently the repo is set up only for classification problems, only allows 2D convolutions, and a number of other restrictions. This should be expanded to handle any kind of structure the user wants with any output and loss function.
- **Code cleanup & structure.** Create a better marginalized, more explicit `GenomeHandler` interface that's cleaner and more extensible. 
- **Expand the search space.** Keras has a fantastic collection of different layers and hyperparameters. Incorporate as many as possible (and sensical) into the GA, while maintaining a simple interface for the user to customize the set of hyperparameters that are permitted. Learning rates, image data augmentation, etc. 
- **Improve the evolution.** GA experts: take this simple approach and improve upon it for better evolution. We may want to use some genetic algorithm library such as deap that's more comprehensive than writing everything by hand.
- **Include in PyPI** for easy pip installation.
- **Parallelize (future).** The task of training multiple nets to evaluate fitness is an incredibly parallelizable task. This is something that should be added after establishing a little more stability.
- **Fine tuning on pre-trained weights (future).** At some point, it would be interesting to provide an interface for users to use GA to find the best fine-tuning approach to their already trained models. This would be especially useful for transfer learning applications.

## Bugs

Anyone can open an issue to report a bug or other issue so long as 1) the issue pertains to DEvol and not Keras, Tensorflow, etc., 2) that there is not already an open issue reported, and 3) that you've pulled the latest changes. Make sure to provide enough information so that the issue can be reproduced. Also, if possible, try fixing it on your own!

## Improvements

See the [projects page](https://github.com/joeddav/devol/projects/1) and read the `vision` to understand tasks you can take on to improve DEvol. If you would like to discuss it or need clarification, open an issue. Alternatively, if there are open issues, grab one of them and tackle it. Also, make sure to use clear, descriptive commit names and maintain a clean history. Then open a merge request to `master` and we'll review it.
