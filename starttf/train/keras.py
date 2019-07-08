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

from hyperparams.hyperparams import import_params, load_params

import starttf
from starttf import PHASE_TRAIN, PHASE_VALIDATION
from starttf.train.params import check_completness
from starttf.data import create_input_fn, buffer_dataset_as_tfrecords
from starttf.utils.plot_losses import create_keras_callbacks
from starttf.utils.create_optimizer import create_keras_optimizer
from starttf.utils.find_loaded_files import get_loaded_files, get_backup_path, copyfile


def rename_fn(fn, name):
    def tmp(*args, **kwargs):
        fn(*args, **kwargs)
    tmp.__name__ = name
    return tmp


def easy_train_and_evaluate(hyperparams, model=None, loss=None, metrics=None,
                            training_data=None, validation_data=None,
                            optimizer=None, epochs=None,
                            continue_training=False, continue_with_specific_checkpointpath=None,
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
    check_completness(hyperparams)
    starttf.hyperparams = hyperparams
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    chkpt_path = hyperparams.train.checkpoint_path + "/" + time_stamp
    chkpt_path = chkpt_path + "_" + hyperparams.train.experiment_name

    if continue_with_specific_checkpointpath:
        chkpt_path = hyperparams.train.checkpoint_path + "/" + continue_with_specific_checkpointpath
        print("Continue with checkpoint: {}".format(chkpt_path))
    elif continue_training:
        chkpts = sorted([name for name in os.listdir(hyperparams.train.checkpoint_path)])
        chkpt_path = hyperparams.train.checkpoint_path + "/" + chkpts[-1]
        print("Latest found checkpoint: {}".format(chkpt_path))

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    # Try to retrieve optional arguments from hyperparams if not specified
    if model is None:
        if isinstance(hyperparams.arch.model, str):
            p = ".".join(hyperparams.arch.model.split(".")[:-1])
            n = hyperparams.arch.model.split(".")[-1]
            arch_model = __import__(p, fromlist=[n])
            model = arch_model.__dict__[n]()
        else:
            model = hyperparams.arch.model()
    if loss is None and hyperparams.arch.get("loss", None) is not None:
        if isinstance(hyperparams.arch.loss, str):
            p = ".".join(hyperparams.arch.loss.split(".")[:-1])
            n = hyperparams.arch.loss.split(".")[-1]
            arch_loss = __import__(p, fromlist=[n])
            loss = arch_loss.__dict__[n]()
        else:
            loss = hyperparams.arch.loss()
    if metrics is None and hyperparams.arch.get("metrics", None) is not None:
        if isinstance(hyperparams.arch.metrics, str):
            p = ".".join(hyperparams.arch.metrics.split(".")[:-1])
            n = hyperparams.arch.metrics.split(".")[-1]
            arch_loss = __import__(p, fromlist=[n])
            metrics = arch_loss.__dict__[n]()
        else:
            metrics = hyperparams.arch.metrics()
    if optimizer is None and hyperparams.train.get("optimizer", None) is not None:
        optimizer, lr_scheduler = create_keras_optimizer(hyperparams)
    if epochs is None:
        epochs = hyperparams.train.get("epochs", 1)

    if training_data is None and validation_data is None:
        augment_train = None
        augment_test = None
        if "augment" in hyperparams.arch.__dict__:
            if isinstance(hyperparams.arch.augment, str):
                p = ".".join(hyperparams.arch.augment.split(".")[:-1])
                n = hyperparams.arch.augment.split(".")[-1]
                arch_augment = __import__(p, fromlist=[n])
                augment = arch_augment.__dict__[n]()
            else:
                augment = hyperparams.arch.augment()
            augment_train = augment.train
            augment_test = augment.test
        if hyperparams.problem.tf_records_path is not None:  # Use tfrecords buffer
            tmp = hyperparams.train.batch_size
            hyperparams.train.batch_size = 1
            buffer_dataset_as_tfrecords(hyperparams)
            hyperparams.train.batch_size = tmp
            training_data, training_samples = create_input_fn(
                hyperparams, PHASE_TRAIN, augmentation_fn=augment_train, repeat=False)
            training_data = training_data()
            validation_data, validation_samples = create_input_fn(
                hyperparams, PHASE_VALIDATION, augmentation_fn=augment_test, repeat=False)
            validation_data = validation_data()
        else:  # Load sequence directly
            if isinstance(hyperparams.arch.prepare, str):
                p = ".".join(hyperparams.arch.prepare.split(".")[:-1])
                n = hyperparams.arch.prepare.split(".")[-1]
                prepare = __import__(p, fromlist=[n])
                prepare = prepare.__dict__[n]
            else:
                prepare = hyperparams.arch.prepare
            training_data = prepare(hyperparams, PHASE_TRAIN, augmentation_fn=augment_train)
            training_samples = len(training_data) * hyperparams.train.batch_size
            validation_data = prepare(hyperparams, PHASE_VALIDATION, augmentation_fn=augment_train)
            validation_samples = len(validation_data) * hyperparams.train.batch_size

    # Write loaded code to output dir
    loaded_files = get_loaded_files()
    for f in loaded_files:
        f_backup = get_backup_path(f, outp_dir=os.path.join(chkpt_path, "src"))  # FIXME create backup path from filepath.
        copyfile(f, f_backup)

    if training_data is not None:
        hyperparams.train.steps = hyperparams.train.epochs * training_samples
    hyperparams.immutable = True

    losses = loss.losses
    metrics = metrics.metrics
    callbacks = create_keras_callbacks(hyperparams, chkpt_path)
    callbacks.append(lr_scheduler)

    # first batches features
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
    model.fit_generator(training_data, validation_data=validation_data, epochs=hyperparams.train.get("epochs", 50),
                        callbacks=callbacks, workers=2, use_multiprocessing=False, shuffle=True, verbose=1)

    return chkpt_path


def main(args):
    if len(args) == 2 or len(args) == 3:
        continue_training = False
        idx = 1
        if args[idx] == "--continue":
            continue_training = True
            idx += 1
        if args[1].endswith(".json"):
            hyperparams = load_params(args[idx])
        elif args[1].endswith(".py"):
            hyperparams = import_params(args[idx])
        setproctitle("train {}".format(hyperparams.train.experiment_name))
        return easy_train_and_evaluate(hyperparams, continue_training=continue_training)
    else:
        print("Usage: python -m starttf.train.keras [--continue] hyperparameters/myparams.py")
        return None


if __name__ == "__main__":
    main(sys.argv)
