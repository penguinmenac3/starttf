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

from hyperparams.hyperparams import load_params

if tf.__version__.startswith("1."):
    print("Using keras for tensorflow 1.x")
    from starttf.train.keras import easy_train_and_evaluate
else:
    from starttf.train.supervised import easy_train_and_evaluate

if len(sys.argv) == 2 or len(sys.argv) == 3:
    continue_training = False
    no_artifacts = False
    idx = 1
    if sys.argv[idx] == "--continue":
        continue_training = True
        idx += 1
    if sys.argv[idx] == "--no_artifacts":
        no_artifacts = True
        idx += 1
    hyperparams = load_params(sys.argv[1])
    name = hyperparams.train.get("experiment_name", "unnamed")
    setproctitle("train {}".format(name))
    easy_train_and_evaluate(hyperparams, continue_training=continue_training, log_suffix=name, no_artifacts=no_artifacts)
else:
    print("Usage: python -m starttf.train [--continue] hyperparameters/myparams.json")
