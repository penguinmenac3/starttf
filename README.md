# starttf - Simplified Deeplearning for Tensorflow [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo aims to contain everything required to quickly develop a deep neural network with tensorflow.
The idea is that if you use write a compatible tf.keras.utils.Sequence for data loading and networks based on the starttf.Module, you will automatically obey best practices and have super fast training speeds.
For writing a keras Sequence the simple way I highly recommend using [opendatalake.SimpleSequence](https://github.com/penguinmenac3/opendatalake/blob/master/opendatalake/simple_sequence.py).

## Install

Properly install `tensorflow` or `tensorflow-gpu` please follow the [official instructions](https://www.tensorflow.org/install/) carefully.

Then, simply pip install from the github repo.

```bash
pip install git+https://github.com/penguinmenac3/starttf.git@tf2
```

## Datasets

Extensions SimpleSequences from [opendatalake.simple_sequence.SimpleSequence](https://github.com/penguinmenac3/opendatalake/blob/master/opendatalake/simple_sequence.py) are supported as well as tf.keras.utils.Sequence (it is important to use the tensorflow keras sequence).
They work like keras.Sequence however with an augmentation and a preprocessing function.

For details checkout the [readme of opendatalake](https://github.com/penguinmenac3/opendatalake/blob/master/README.md).

## Module

A module is a class which has state to keep variables and can be called like a function.
The function call parameters and returns are arbitrary and depend on the implementation of `def call(self, ...):`

### Model Module

To use a module as a model it must have a `training`-parameter and return it's outputs as a dictionary or namedtuple.
Note that the names must match the names of the sequence outputs and the loss function names.

### Loss Module

To use a module as a loss it must have accept `y_true` and `y_pred` as two parameters in this order.
It must return a the loss as a single scalar tensor as return of the call method.
Furthermore it has to store all it's partial losses in a dict in self.losses with the keys matching those of the model and the dataset in order to work with tf1 and keras.

## Simple to use tensorflow

### Simple Training (No Boilerplate)

When you do standard supervised training, you do not need to write your own trainer code.
Simply call the library with the hyperparams filepath as a parameter.

```bash
python -m starttf.train myhyperparams.json
```

If you need a custom training method, you can find inspiration in the tf2 training implemented [here](starttf/train/supervised.py).

### Quick Model Definition

Simply implement a create_model function.
This model is only a feed forward model.

The model function returns a dictionary containing all layers that should be accessible from outside and a dictionary containing debug values that should be availible for loss or plotting in tensorboard.

```python
import tensorflow as tf

from starttf.modules import Module
from starttf.modules.encoders import Encoder

Conv2D = tf.keras.layers.Conv2D


class ExampleModel(Module):
    def __init__(self):
        super(ExampleModel, self).__init__(name=ExampleModel)
        num_classes = self.hyperparams.problem.number_of_categories

        # Create the vgg encoder
        self.encoder = Encoder()

        #Use the generated model 
        self.conv6 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same", activation="relu")
        self.conv7 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same", activation="relu")
        self.conv8 = Conv2D(filters=num_classes, kernel_size=(1, 1), padding="same", activation=None, name="probs")

    def call(self, image, training=False):
        """
        Run the model.
        """
        features, debug = self.encoder(image, training)
        result = self.conv6(features)
        result = self.conv7(result)
        logits = self.conv8(result)
        probs = tf.nn.softmax(logits)
        return {"logits": logits, "probs": probs}
```

### Quick Loss Definition

An example loss as often used in 2d object detection.

```python
import tensorflow as tf

from starttf.modules import CompositeLoss
from starttf.losses.basic_losses import smooth_l1_distance
from starttf.losses.loss_processors import mask_loss


class Detection2DLoss(CompositeLoss):
    def __init__(self, name="Detection2DLoss"):
        super().__init__(name=name)
        k = self.hyperparams.problem.number_of_categories

        def class_loss(y_true, y_pred):
            '''Just another crossentropy'''
            with tf.name_scope("class_loss"):
                mask = tf.cast(tf.not_equal(y_true, k), tf.float32)[:, :, :,0]
                y_true = tf.cast(y_true, tf.uint8)

                one_hot = tf.one_hot(y_true, k+1)
                shape = one_hot.get_shape().as_list()
                y_true = tf.reshape(one_hot, shape=[-1, shape[1], shape[2], shape[4]])

                loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true[:, :, :, :k])
                return tf.reduce_mean(mask_loss(input_tensor=loss, binary_tensor=mask))

        def regression_loss(y_true, y_pred):
            '''Just another crossentropy'''
            with tf.name_scope("rect_loss"):
                # Use height as a mask, since it cannot be 0.
                mask = tf.cast(tf.not_equal(y_true[:, :, :, 3], 0), tf.float32)

                l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
                loss = smooth_l1_distance(labels=y_true, preds=y_pred)
                return tf.reduce_mean(mask_loss(input_tensor=loss, binary_tensor=mask))

        self.losses = {"class_logits": class_loss, "rect": regression_loss}
        self.metrics = {}

```

### Tensorboard Integration

Tensorboard integration is simple.

Every loss in the losses dict is automatically added to tensorboard. If you also want debug images or an extra scalar summary, you can add a module.summary(summary_name=image) in your modules call-method.

TODO (currently not implemented)

### TF Estimator + Cluster Support

If you use the easy_train_and_evaluate method, a correctly configured TF Estimator is created.
The estimator is then trained in a way that supports cluster training if you have a cluster.

TODO (currently not implemented)
