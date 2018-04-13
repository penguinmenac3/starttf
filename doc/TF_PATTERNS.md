# Good Patterns for Deep Learning with Tensorflow

This folder describes several patterns for tensorflow I found usefull.
Besides describing patterns, I will try to explain why they could be usefull.

All patterns here are basically what lead the implementation of this repository.

## Patterns

(Description what the patterns mean and why below.)
1. Functional Programming Style
2. Separate Model, Loss & Optimizer
3. TF-Records or at least Input queues (no feed dict!)
4. Hyperparameter file (not arguments, don't even think about constants in code!)
5. Train and eval model (reuse_weights)
6. Summaries and Matplotlib metrics

## Functional Programming Style

Model, loss and training should not be implemented as classes but rather as functions.
These functions should be side effect free (appart from reuse variables).

Why?
1. A neural network is a function approximator and this keeps your code in sync with the thought process about the model. And creating advanced models is basically sticking together functions and their results. (e.g. y = vgg(img), class = classifier_head(y))
2. Mix & Match. This is means sticking together multiple encoder and decoder combinations easily.
If you have for example a vgg or a googlenet encoder, you can easily swap them out.
The same goes for decoders for classification, bounding box regression or segmentation and even the loss and optimizer.
This simplifies reusability of code, since simply the inputs and outputs must match and there is no state to worry about.
3. Side effect free code. This actually enables the above point to be possible (without worries).
4. Loss, network and data independant.
5. Easy to import/wrap other models (simply put it in a function)
6. tf.Estimator compatible (see model_fn)

## Separate Model, Loss and Optimizer

Have a separate function that creates the model. Even have a separate function for loss and optimizer.
This greatly speeds up development and reduces code complexity.

Your functions will get alot smaller and therfore easier to understand.

This gives the ability to rapidly develop different models on the same data and loss without duplicating code.
On the otherside it also gives the ability to reuse a model on different tasks and datasets.

For debugging it can be handy to have a simple (low quality) network to check if your error is in the loss specification or the network architecture itself.

## TF-Records or at least Input queues (no feed dict!)

To feed data into your network do not use a feed dict, but tfrecords or at least input queues filled by multiple threads.

Why? Performance. It is so much faster.

## Hyperparameter file (not arguments)

Do not use constants or magic variables in your code, put them in a hyperparameter file.
Copy the training file into your checkpoint folder at the start of every training.

Many other people use arguments for their variables such as 'batch_size', 'learning_rate' etc.
However, I recommend not following this trend, because when you do lot's of experiments it is easy to loose track what parameters where used for what experiment.
Having a copy of the hyperparameter file alongside the checkpoints and the loss plots, helps reconstructing what you did.

## Train and eval model (reuse_weights)

During training you want to have a training model and an eval model that reuses the training models weights.

You need the model twice, since the training and the eval model will have differenecs.
One common would be dropout or simply the input and output tensors.

One reason for both models might be that you should have a training loss as well as a validation loss.
Another more subtle one might be speed.

## TFRecords and Matplotlib metrics

Use Summaries provided by tensorflow alongside custom matplotlib plots.

Probably everyone knows why to use summaries for tensorboard.
It is a nice way of tracking your training process in the browser and get insights in what your network is doing.

However, when writing a paper/thesis/documentation you should not use tensorboard plots.
And for that it is handy to have matplotlib plots of your training process.

And there is another thing.
When training with the pattern to have a train and eval model, their loss curves do not show up in the same plot in tensorboard.
To detect if you have generalization problems is by far simpler by having both curves in a single plot.
When implementing your own matplotlib plots you can achieve that.
