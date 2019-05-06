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


#@tf.function
def __train(model, dataset, optimizer, loss_fn):
    i = 0
    N = len(dataset)
    for x, y in dataset:
        tf.keras.backend.set_learning_phase(1)
        with tf.GradientTape() as tape:
            prediction = model(**x)
            loss = loss_fn(y, prediction)
        tf.keras.backend.set_learning_phase(0)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(gradients, model.trainable_variables)
        print("\rBatch {}/{} - {}".format(i+1, N, loss), end="")
        i += 1

#@tf.function
def __eval(model, dataset, eval_fn):
    total_loss = 0
    for x, y in dataset:
        prediction = model(x, training=False)
        total_loss += eval_fn(y, prediction)
    return total_loss / len(dataset)

def easy_train_and_evaluate(hyperparams, model=None, loss=None, evaluator=None,
                            training_data=None, validation_data=None,
                            optimizer=None, epochs=None,
                            continue_training=False, continue_with_specific_checkpointpath=None, no_artifacts=False,
                            train_fn=__train, eval_fn=__eval, create_optimizer=create_keras_optimizer):
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

    if not os.path.exists(chkpt_path) and not no_artifacts:
        os.makedirs(chkpt_path)

    # TODO setup tensorboard logging and checkpoints as well as model saving (when not no_artifacts)

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
    if evaluator is None and hyperparams.arch.get("eval", None) is not None:
        p = ".".join(hyperparams.arch.eval.split(".")[:-1])
        n = hyperparams.arch.eval.split(".")[-1]
        arch_loss = __import__(p, fromlist=[n])
        evaluator = arch_loss.__dict__[n]()
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
    if model is None or loss is None or evaluator is None or training_data is None or validation_data is None or optimizer is None or epochs is None:
        raise RuntimeError("You must provide all arguments either directly or via hyperparams.")

    print("Epoch {}/{}".format(1, epochs))
    for i in range(epochs):
        train_fn(model, training_data, optimizer, loss)
        score = eval_fn(model, validation_data, evaluator)
        print("\rEpoch {}/{} - {}".format(i+1, epochs, score))

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
