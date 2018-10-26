# starttf - Simplified Deeplearning for Tensorflow [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo aims to contain everything required to quickly develop a deep neural network with tensorflow.
The idea is that if you use write a compatible SimpleSequence for data loading and networks based on the StartTFModel, you will automatically obey best practices and have super fast training speeds.

## Install

Properly install `tensorflow` or `tensorflow-gpu` please follow the [official instructions](https://www.tensorflow.org/install/) carefully.

Then, simply pip install from the github repo.

```bash
pip install starttf
```

## Datasets

Extensions SimpleSequences from [opendatalake.simple_sequence.SimpleSequence](https://github.com/penguinmenac3/opendatalake/blob/master/opendatalake/simple_sequence.py) are supported.
They work like keras.Sequence however with an augmentation and a preprocessing function.

For details checkout the [readme of opendatalake](https://github.com/penguinmenac3/opendatalake/blob/master/README.md).

## Models

Every model returns a dictionary containing output tensors and a dictionary containing debug tensors

1. [Model Base Classes](starttf/models/models.py)
2. [Common Encoders](starttf/models/encoders.py)
3. [Untrained Backbones](starttf/models/backbones)

## Simple to use tensorflow

### Simple Training (No Boilerplate)

There are pre-implemented models which can be glued together and trained with just a few lines.
However, before training you will have to create tf-records as shown in the section *Simple TF Record Creation*.
This is actually a full main file.

```python
# Import helpers
from starttf.estimators.tf_estimator import easy_train_and_evaluate
from starttf.utils.hyperparams import load_params

# Import a/your model (here one for mnist)
from mymodel import MyStartTFModel

# Import your loss (here an example)
from myloss import create_loss

# Load params (here for mnist)
hyperparams = load_params("hyperparams/experiment1.json")

# Train model
easy_train_and_evaluate(hyperparams, MyStartTFModel, create_loss, continue_training=False)
```

### Quick Model Definition

Simply implement a create_model function.
This model is only a feed forward model.

The model function returns a dictionary containing all layers that should be accessible from outside and a dictionary containing debug values that should be availible for loss or plotting in tensorboard.

```python
import tensorflow as tf

from starttf.models.model import StartTFModel
from starttf.models.encoders import Encoder

Conv2D = tf.keras.layers.Conv2D


class ExampleModel(StartTFModel):
    def __init__(self, hyperparams):
        super(ExampleModel, self).__init__(hyperparams)
        num_classes = hyperparams.problem.number_of_categories

        # Create the vgg encoder
        self.encoder = Encoder(hyperparams)

        #Use the generated model 
        self.conv6 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same", activation="relu")
        self.conv7 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same", activation="relu")
        self.conv8 = Conv2D(filters=num_classes, kernel_size=(1, 1), padding="same", activation=None, name="probs")

    def call(self, input_tensor, training=False):
        """
        Run the model.
        """
        encoder, debug = self.encoder(input_tensor, training)
        result = self.conv6(encoder["features"])
        result = self.conv7(result)
        logits = self.conv8(result)
        probs = tf.nn.softmax(logits)
        return {"logits": logits, "probs": probs}, debug
```

### Quick Loss Definition

```python
def create_loss(model, labels, mode, hyper_params):
    metrics = {}
    losses = {}

    # Add loss
    labels = tf.reshape(labels["probs"], [-1, hyper_params.problem.number_of_categories])
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model["logits"], labels=labels)
    loss_op = tf.reduce_mean(ce)

    # Add losses to dict. "loss" is the primary loss that is optimized.
    losses["loss"] = loss_op
    metrics['accuracy'] = tf.metrics.accuracy(labels=labels,
                                              predictions=model["probs"],
                                              name='acc_op')

    return losses, metrics
```

### Simple TF Record Creation

Fast training speed can be achieved by using tf records.
However, usually tf records are a hastle to use the write_data method makes it simple.

```python
from starttf.utils.hyperparams import load_params
from starttf.data.autorecords import write_data

from my_data import MySimpleSequence

# Load the hyper parameters.
hyperparams = load_params("hyperparams/experiment1.json")

# Get a generator and its parameters
training_data = MySimpleSequence(hyperparams)
validation_data = MySimpleSequence(hyperparams)

# Write the data
write_data(hyperparams, PHASE_TRAIN, training_data, 4)
write_data(hyperparams, PHASE_VALIDATION, validation_data, 2)
```

### Tensorboard Integration

Tensorboard integration is simple.

Every loss in the losses dict is automatically added to tensorboard.
If you also want debug images, you can add a tf.summary.image() in your create_loss method.

### TF Estimator + Cluster Support

If you use the easy_train_and_evaluate method, a correctly configured TF Estimator is created.
The estimator is then trained in a way that supports cluster training if you have a cluster.
