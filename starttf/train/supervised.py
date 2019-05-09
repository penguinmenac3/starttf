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
from hyperparams.hyperparams import import_params, load_params
import starttf
from starttf.train.params import check_completness
from starttf.utils.create_optimizer import create_keras_optimizer


PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


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


# @tf.function
def __train(model, dataset, optimizer, loss, metrics, step):
    i = 0
    N = len(dataset)
    tf.keras.backend.set_learning_phase(1)
    for x, y in dataset:
        with tf.GradientTape() as tape:
            prediction = model(**x)
            loss_results = loss(y, prediction)
            metrics(y, prediction)
        gradients = tape.gradient(loss_results, model.trainable_variables)
        optimizer.apply_gradients(gradients, model.trainable_variables)
        print("\rBatch {}/{} - {} - {}".format(i + 1, N, __dict_to_str(loss.values), __dict_to_str(metrics.values)), end="")
        if i % starttf.hyperparams.train.log_steps == 0:
            for k in loss.values:
                tf.summary.scalar("loss/{}".format(k), loss.values[k],
                                  step=step + i * starttf.hyperparams.train.batch_size)
            for k in metrics.values:
                tf.summary.scalar("metrics/{}".format(k),
                                  metrics.values[k], step=step + i * starttf.hyperparams.train.batch_size)
        i += 1
    tf.keras.backend.set_learning_phase(0)


# @tf.function
def __validate(model, dataset, loss, metrics, step):
    tf.keras.backend.set_learning_phase(0)
    for x, y in dataset:
        prediction = model(**x)
        loss(y, prediction)
        metrics(y, prediction)
    for k in loss.avg:
        tf.summary.scalar("loss/{}".format(k), loss.avg[k],
                          step=step)
    for k in metrics.avg:
        tf.summary.scalar("metrics/{}".format(k),
                          metrics.avg[k], step=step)
    return loss.avg, metrics.avg


def easy_train_and_evaluate(hyperparams, model=None, loss=None, metrics=None,
                            training_data=None, validation_data=None,
                            optimizer=None, epochs=None,
                            continue_training=False, continue_with_specific_checkpointpath=None,
                            train_fn=__train, validation_fn=__validate, create_optimizer=create_keras_optimizer):
    hyperparams.immutable = True
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

    if not os.path.exists(chkpt_path + "/train"):
        os.makedirs(chkpt_path + "/train")
    if not os.path.exists(chkpt_path + "/val"):
        os.makedirs(chkpt_path + "/val")

    # Summary writers
    train_summary_writer = tf.summary.create_file_writer(chkpt_path + "/train")
    val_summary_writer = tf.summary.create_file_writer(chkpt_path + "/val")

    # Try to retrieve optional arguments from hyperparams if not specified
    if model is None:
        p = ".".join(hyperparams.arch.model.split(".")[:-1])
        n = hyperparams.arch.model.split(".")[-1]
        arch_model = __import__(p, fromlist=[n])
        model = arch_model.__dict__[n]()
    if loss is None and hyperparams.arch.get("loss", None) is not None:
        p = ".".join(hyperparams.arch.loss.split(".")[:-1])
        n = hyperparams.arch.loss.split(".")[-1]
        arch_loss = __import__(p, fromlist=[n])
        loss = arch_loss.__dict__[n]()
    if metrics is None and hyperparams.arch.get("metrics", None) is not None:
        p = ".".join(hyperparams.arch.metrics.split(".")[:-1])
        n = hyperparams.arch.metrics.split(".")[-1]
        arch_loss = __import__(p, fromlist=[n])
        metrics = arch_loss.__dict__[n]()
    if training_data is None and hyperparams.arch.get("prepare", None) is not None:
        p = ".".join(hyperparams.arch.prepare.split(".")[:-1])
        n = hyperparams.arch.prepare.split(".")[-1]
        prepare = __import__(p, fromlist=[n])
        prepare = prepare.__dict__[n]
        training_data = prepare(hyperparams, PHASE_TRAIN)
        validation_data = prepare(hyperparams, PHASE_VALIDATION)
    if optimizer is None and hyperparams.train.get("optimizer", None) is not None:
        optimizer, lr_scheduler = create_optimizer(hyperparams)
    if epochs is None:
        epochs = hyperparams.train.get("epochs", 1)

    # Check if all requirements could be retrieved.
    if model is None or loss is None or metrics is None or training_data is None or validation_data is None or optimizer is None or epochs is None:
        raise RuntimeError("You must provide all arguments either directly or via hyperparams.")

    # Load Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, chkpt_path, max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)

    print("Epoch {}/{}".format(1, epochs))
    for i in range(epochs):
        step = i * len(training_data) * starttf.hyperparams.train.batch_size
        loss.reset()
        metrics.reset()
        with train_summary_writer.as_default():
            train_fn(model, training_data, optimizer, loss, metrics, step)
        step = (i + 1) * len(training_data) * starttf.hyperparams.train.batch_size
        loss.reset()
        metrics.reset()
        with val_summary_writer.as_default():
            loss_results, metrics_results = validation_fn(model, validation_data, loss, metrics, step)

        ckpt.step.assign_add(1)
        save_path = manager.save()
        print("\nEpoch {}/{} - {} - {}".format(i + 1, epochs, __dict_to_str(loss_results), __dict_to_str(metrics_results)))

    return chkpt_path


def main(args):
    if len(args) == 2 or len(args) == 3:
        continue_training = False
        no_artifacts = False
        idx = 1
        if args[idx] == "--continue":
            continue_training = True
            idx += 1
        if args[idx] == "--no_artifacts":
            no_artifacts = True
            idx += 1
        if args[1].endswith(".json"):
            hyperparams = load_params(args[idx])
        elif args[1].endswith(".py"):
            hyperparams = import_params(args[idx])
        setproctitle("train {}".format(hyperparams.train.experiment_name))
        return easy_train_and_evaluate(hyperparams, continue_training=continue_training, no_artifacts=no_artifacts)
    else:
        print("Usage: python -m starttf.train.supervised [--continue] hyperparameters/myparams.py")
        return None


if __name__ == "__main__":
    main(sys.argv)
