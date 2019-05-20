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

from hyperparams import HyperParams as OriginalParams


def __has_attribute(obj, name):
    return name in obj.__dict__ and obj.__dict__[name] is not None


def check_completness(params):
    # Check for training parameters
    assert __has_attribute(params, "train")
    assert __has_attribute(params.train, "experiment_name")
    assert __has_attribute(params.train, "checkpoint_path")
    assert __has_attribute(params.train, "batch_size")
    assert __has_attribute(params.train, "learning_rate")
    assert __has_attribute(params.train.learning_rate, "type")
    assert __has_attribute(params.train.learning_rate, "start_value")
    if params.train.learning_rate.type == "exponential":
        assert __has_attribute(params.train.learning_rate, "end_value")
    assert __has_attribute(params.train, "optimizer")
    assert __has_attribute(params.train.optimizer, "type")
    assert __has_attribute(params.train, "epochs")

    assert __has_attribute(params, "arch")
    assert __has_attribute(params.arch, "model")
    assert __has_attribute(params.arch, "loss")
    assert __has_attribute(params.arch, "metrics")
    assert __has_attribute(params.arch, "prepare")

    assert __has_attribute(params, "problem")
    #assert __has_attribute(params.problem, "tf_records_path")


class HyperParams(OriginalParams):
    def __init__(self, d=None):
        self.train = OriginalParams()

        self.train.batch_size = 1
        self.train.experiment_name = None
        self.train.checkpoint_path = "checkpoints"
        self.train.epochs = 50
        self.train.log_steps = 100
        self.train.learning_rate = OriginalParams()
        self.train.learning_rate.type = "const"
        self.train.learning_rate.start_value = 0.001
        self.train.learning_rate.end_value = 0.0001
        self.train.optimizer = OriginalParams()
        self.train.optimizer.type = "adam"

        self.arch = OriginalParams()
        self.arch.model = None
        self.arch.loss = None
        self.arch.eval = None
        self.arch.prepare = None

        self.problem = OriginalParams()
        self.problem.base_dir = None
        self.problem.tf_records_path = None

        super().__init__(d=d)
