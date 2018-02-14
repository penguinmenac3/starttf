# Deep Learning - Starterpack [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo aims to contain everything required to quickly develop a deep neural network with tensorflow or keras.
It comes with several dataset loaders and network architectures.
If I can find it the models will also contain pretrained weights.
The idea is that if you use existing dataset loaders and networks and only modify them, you will automatically obey best practices and have super fast training speeds.

## Install

First clone the repository **recursively**!

```bash
git clone --recursive https://github.com/penguinmenac3/deeplearning-starterpack.git
```

Simply create a new python virtual environment (preferably python 3.x) and install the requirements listed in the [requirements.txt](requirements.txt).
Properly install tensorflow-gpu please follow the [official instructions](https://www.tensorflow.org/install/) carefully.


## Running examples

Either use an ide such as pycharm and set your working directory to the keras-starterpack folder.
Or launch your code from the command line like the example bellow.

```bash
# Activate your virtual environment (in my case venv)
# Then do the following
(venv) $ cd /PATH/TO/deeplearning-starterpack
(venv) $ pyhon -m examples.mnist
```

## Datasets

For dataset support I use my own dataset library including bindings to load many popular datasets in a unified format.

However, to use it you will need to initialize git submodules if you did not do a recursive clone:

```bash
git submodule update --init --recursive
```

The dataset loader basically supports **classification**, **segmentation**, **regression** (including **2d- and 3d-detection**) and some visualization helpers.
For details checkout the readme of the project [**here**](https://github.com/penguinmenac3/datasets/blob/master/README.md).

## Models

There are some models implemented to tinker around with.
Most of the implementations are not done by me from scratch but rather refactoring of online found implementations.
Also the common models will come with pre trained weights I found on the internet.
Just check the comment at the top of their source files.

### Tensorflow Models

Every [model](tf_models/model.py) supports setup, predict, fit and export methods.

1. Alexnet (single stream version) [TODO]
2. VGG 16 [TODO]
3. GoogLeNet (Inception v3) [TODO]
4. Overfeat/Tensorbox [TODO]
5. ResNet [TODO]
6. SegNet [TODO]
7. Mask RCNN [TODO]
8. monoDepth [TODO]

More non famous models by myself:

1. [CNN for MNIST (Digit Recognition)](tf_models/mnist.py)
2. [GRU Function Classifier](tf_models/gru_function_classifier.py)
3. CNN for LFW (Person Identification) [TODO]

### Keras Models

1. [Alexnet (single stream version)](keras_models/alexnet.py)
2. [VGG 16](keras_models/vgg_16.py)
3. [GoogLeNet (Inception v3)](keras_models/googlenet.py)
4. Overfeat/Tensorbox [TODO]
5. ResNet [TODO]
6. [SegNet](keras_models/segnet.py)
7. Mask RCNN [TODO]
8. monoDepth [TODO]

More non famous models by myself:

1. [CNN for MNIST](keras_models/mnist_cnn.py)
2. [CNN for Person Classification](keras_models/tinypersonnet.py)
3. CNN for Person Identification [WIP]

## Examples

Some samples that should help getting into stuff.

### Tensorflow Examples

1. [MNIST](tf_examples/mnist.py)
2. LFW [WIP]
3. Imagenet (Baselines) [TODO]
4. Bounding Box Regression [TODO]
5. Segmentations [TODO]
6. Instance Masks [TODO]
7. Reinforcement Learning [TODO]
8. [GRU Function Classifier](tf_examples/gru_function_classifier.py)

### Keras Examples


Notebooks:
1. [MNIST Notebook](keras_examples/mnist.ipynb)

Code:

1. [MNIST](keras_examples/mnist.py)
2. LFW [WIP]
3. Imagenet (Baselines) [TODO]
4. Bounding Box Regression [TODO]
5. Segmentations [TODO]
6. Instance Masks [TODO]
7. Reinforcement Learning [TODO]

On non publically availible data:
(however can be used on your own data)

1. [Simple Classification (Folder per Class)](keras_examples/tinypersonnet.py)


## Simple to use tensorflow


### Predefined Models

There are pre-implemented models. Simply import them and link them to your session.

```python
from tf_models.mnist import Mnist
# Create Model
model = Mnist(hyper_params_filepath)

# Create a session and link it to your model.
with tf.Session(config=config) as sess:
    model.setup(sess)
```

### Quick Model Definition

Simply implement the _create_model method in a derived class from Model to define your very own model.
Just use a variable scope and reuse weights if requested from outside and return outputs.

```python
from tf_models.model import Model
class YourModel(Model):
    # [...]

    def _create_model(self, input_tensor, reuse_weights, is_deploy_model=False):
        outputs = {}
        with tf.variable_scope('MnistNetwork') as scope:
            if reuse_weights:
                scope.reuse_variables()

            # TODO put your network here

            outputs["logits"] = logits
            outputs["probs"] = probs
        return outputs
```

Now you can define your loss for training. In the case of this is super simple.
The output returned from the _create_model method are now available as class variables in self.model_train and self.model_deploy for you.

The train model is meant to be used for training (optimizing weights) whereas the deploy model should be used for evaluation/validation of your model.

```python
    def _create_loss(self, labels, validation_labels=None):
        labels = tf.reshape(labels, [-1, 10])
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model_train["logits"], labels=labels))
        train_op = tf.train.RMSPropOptimizer(learning_rate=self.hyper_params.train.learning_rate, decay=self.hyper_params.train.decay).minimize(loss_op)
        tf.summary.scalar('train/loss', loss_op)

        # Create a validation loss if possible.
        validation_loss_op = None
        if validation_labels is not None:
            validation_labels = tf.reshape(validation_labels, [-1, 10])
            validation_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model_deploy["logits"], labels=validation_labels))
            tf.summary.scalar('dev/loss', validation_loss_op)

        return train_op, loss_op, validation_loss_op
```

### TFRecord Integration

Fast training speed can be achieved by using tf records.
Actually the api only supports using tf records, to enforce usage for optimal performance.

```python
# Imports
from datasets.classification.mnist import mnist
from datasets.tfrecords import write_tf_records, read_tf_records, PHASE_TRAIN, PHASE_VALIDATION

if data_needs_generation:
    # Load the dataset you want as a generator.
    train_data = mnist(base_dir=base_dir, phase=PHASE_TRAIN)
    validation_data = mnist(base_dir=base_dir, phase=PHASE_VALIDATION)

    # Write a record with 4 threads for training and 2 threads for validation.
    write_tf_records(data_tmp_folder, 4, 2, train_data, validation_data)

# Create your model to know hyperparameters for reader.
model = ...

# Load data with tf records.
train_features, train_labels = read_tf_records(data_tmp_folder, PHASE_TRAIN, model.hyper_params.train.batch_size)

model.fit(train_features, train_labels)  # optional: also pass in validation features and labels
```

### Tensorboard Integration

Tensorboard integration is simple.
You just have to define a summary (e.g. a summary scalar for the loss) and it gets added to the tensorboard.
No worries when to summarize and how to call it and merging.
Simply define your summary and the rest is handled by the meta model in the fit method.

![Screenshot showing code to include tensorboard on the left and tensorboard on the right](images/tensorboard_integration.png)

### More details

More details can be found in the tf_examples or tf_models. Mnist is a simple example for starters.
