# MIT License
#
# Copyright (c) 2018-2019 Michael Fuerst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
import datetime
import json
import sys
from setproctitle import setproctitle

import tensorflow as tf
from tensorflow.keras.models import load_model
import ailab
from ailab import PHASE_TRAIN, PHASE_VALIDATION

import starttf
from starttf.utils.plot_losses import create_keras_callbacks
from starttf.utils.create_optimizer import create_keras_optimizer


def rename_fn(fn, name):
    def tmp(*args, **kwargs):
        fn(*args, **kwargs)
    tmp.__name__ = name
    return tmp


def fit(config, model=None, loss=None, metrics=None,
        training_data=None, validation_data=None,
        optimizer=None,
        continue_training=False, continue_with_specific_checkpointpath=False,
        create_optimizer=create_keras_optimizer):
    """
    Train and evaluate your model without any boilerplate code.

    1) Write your data using the starttf.tfrecords.autorecords.write_data method.
    2) Create your hyper parameter file containing all required fields and then load it using
        starttf.utils.hyper_params.load_params method.
        Minimal Sample Hyperparams File:
        {"train": {
            "learning_rate": {
                "type": "const",
                "start_value": 0.001
            },
            "optimizer": {
                "type": "adam"
            },
            "batch_size": 1024,
            "iters": 10000,
            "summary_iters": 100,
            "checkpoint_path": "checkpoints/mnist",
            "tf_records_path": "data/.records/mnist"
            }
        }
    3) Pass everything required to this method and that's it.
    :param hyper_params: The hyper parameters obejct loaded via starttf.utils.hyper_params.load_params
    :param Model: A keras model.
    :param create_loss: A create_loss function like that in starttf.examples.mnist.loss.
    :param inline_plotting: When you are using jupyter notebooks you can tell it to plot the loss directly inside the notebook.
    :param continue_training: Bool, continue last training in the checkpoint path specified in the hyper parameters.
    :param session_config: A configuration for the session.
    :return:
    """
    config.check_completness()
    chkpt_path = setup_logging(config, continue_with_specific_checkpointpath, continue_training)

    # Try to retrieve optional arguments from hyperparams if not specified
    if model is None:
        model = config.arch.model()
    if loss is None and config.arch.get("loss", None) is not None:
        loss = config.arch.loss()
    if metrics is None and config.arch.get("metrics", None) is not None:
        metrics = config.arch.metrics()
    if optimizer is None and config.train.get("optimizer", None) is not None:
        optimizer, lr_scheduler = create_keras_optimizer(config)
    epochs = config.train.get("epochs", 1)

    training_data, training_samples, validation_data, validation_samples = auto_setup_data(config)

    if training_data is not None:
        config.train.steps = config.train.epochs * training_samples
    config.immutable = True

    losses = loss.losses
    metrics = metrics.metrics
    callbacks = create_keras_callbacks(config, chkpt_path)
    callbacks.append(lr_scheduler)

    # first batches features
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
    model.fit_generator(training_data, validation_data=validation_data, epochs=config.train.get("epochs", 50),
                        callbacks=callbacks, workers=2, use_multiprocessing=False, shuffle=True, verbose=1)

    return chkpt_path
