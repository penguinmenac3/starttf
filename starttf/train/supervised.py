# MIT License
#
# Copyright (c) 2019 Michael Fuerst
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

import sys
import os
import time
import datetime
from setproctitle import setproctitle
import tensorflow as tf
import ailab
from ailab import PHASE_TRAIN, PHASE_VALIDATION
from ailab.experiment import import_config
from ailab.experiment import setup_logging
from ailab.data.tfrecord import write_data, create_input_fn, auto_setup_data

import starttf as stf
from starttf.utils.create_optimizer import create_keras_optimizer


def __dict_to_str(data):
    out = []
    for k in data:
        if isinstance(data[k], list):
            for i in data[k]:
                name = i.__name__
                if isinstance(i, tf.Module):
                    name = i.name
                out.append("{}_{}={:.3f}".format(k, name, data[k].numpy()))
        else:
            out.append("{}={:.3f}".format(k, data[k].numpy()))
    return " - ".join(out)


def format_time(t):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


#@tf.function
def _train(config, model, dataset, samples_per_epoch, optimizer, loss, metrics):
    i = 0
    N = int(samples_per_epoch / config.train.batch_size - 0.00001) + 1
    
    # Setup the training loop
    tf.keras.backend.set_learning_phase(1)
    loss.reset_avg()
    metrics.reset_avg()

    # Loop over the dataset and update weights.
    for x, y in dataset:
        # Forward pass, computing gradients and applying them
        with tf.GradientTape() as tape:
            prediction = model(**x)
            loss_results = loss(y, prediction)
            metrics(y, prediction)
        variables = model.trainable_variables
        gradients = tape.gradient(loss_results, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        # Update global variables and log the variables
        stf.train.samples_seen = stf.train.samples_seen + config.train.batch_size
        # FIXME TypeError: unsupported format string passed to Tensor.__format__
        print("\rBatch {}/{} - Loss {:.3f}".format(i + 1, N, loss_results), end="")
        if i % config.train.log_steps == 0:
            tf.summary.scalar('learning_rate', optimizer.lr, step=stf.train.samples_seen)
            loss.summary()
            metrics.summary()
        i += 1
    tf.keras.backend.set_learning_phase(0)


#@tf.function
def _validate(config, model, dataset, samples_per_epoch, loss, metrics):
    tf.keras.backend.set_learning_phase(0)    
    samples = 0
    for x, y in dataset:
        prediction = model(**x)
        loss(y, prediction)
        metrics(y, prediction)
        samples += config.train.batch_size
        if samples >= samples_per_epoch:
            break
    loss.summary()
    metrics.summary()
    return loss.avg, metrics.avg


def fit(config, model=None, loss=None, metrics=None,
        training_data=None, validation_data=None,
        optimizer=None,
        continue_training=False, continue_with_specific_checkpointpath=False,
        train_fn=_train, validation_fn=_validate, create_optimizer=create_keras_optimizer):
    config.check_completness()
    chkpt_path = setup_logging(config, continue_with_specific_checkpointpath, continue_training)

    # Summary writers
    train_summary_writer = tf.summary.create_file_writer(chkpt_path + "/train")
    val_summary_writer = tf.summary.create_file_writer(chkpt_path + "/val")

    # Try to retrieve optional arguments from hyperparams if not specified
    if model is None:
        model = config.arch.model()
    if loss is None and config.arch.get("loss", None) is not None:
        loss = config.arch.loss()
    if metrics is None and config.arch.get("metrics", None) is not None:
        metrics = config.arch.metrics()
    if optimizer is None and config.train.get("optimizer", None) is not None:
        optimizer, lr_scheduler = create_optimizer(config)
    epochs = config.train.get("epochs", 1)
    model.optimizer = optimizer
    lr_scheduler.model = model
    
    training_data, training_samples, validation_data, validation_samples = auto_setup_data(config)

    # Load Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(chkpt_path, "checkpoints"), max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)

    print("Epoch {}/{}".format(1, epochs))
    stf.train.samples_seen = 0
    start = time.time()
    for i in range(epochs):
        lr_scheduler.on_epoch_begin(i)
        loss.reset_avg()
        metrics.reset_avg()

        with train_summary_writer.as_default():
            train_fn(config, model, training_data, training_samples, optimizer, loss, metrics)
        
        loss.reset_avg()
        metrics.reset_avg()
        with val_summary_writer.as_default():
            loss_results, metrics_results = validation_fn(config, model, validation_data, validation_samples, loss, metrics)

        lr_scheduler.on_epoch_end(i)
        ckpt.step.assign_add(1)
        save_path = manager.save()
        elapsed_time = time.time() - start
        eta = elapsed_time / (i + 1) * (epochs - (i + 1))
        print("\rEpoch {}/{} - ETA {} - {} - {}".format(i + 1, epochs, format_time(eta),
                                                        __dict_to_str(loss_results), __dict_to_str(metrics_results)))

    return chkpt_path
