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
from setproctitle import setproctitle
import tensorflow as tf

from hyperparams.hyperparams import import_params, load_params
from starttf.train.supervised import easy_train_and_evaluate


def main(args):
    if len(args) == 2 or len(args) == 3:
        continue_training = False
        no_artifacts = False
        idx = 1
        if args[idx] == "--continue":
            continue_training = True
            idx += 1
        if args[1].endswith(".json"):
            hyperparams = load_params(args[idx])
        elif args[1].endswith(".py"):
            hyperparams = import_params(args[idx])
        name = hyperparams.train.get("experiment_name", "unnamed")
        setproctitle("train {}".format(name))
        return easy_train_and_evaluate(hyperparams, continue_training=continue_training)
    else:
        print("Usage: python -m starttf.train [--continue] hyperparameters/myparams.py")
        return None
