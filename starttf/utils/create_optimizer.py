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

import math
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler


def create_keras_optimizer(params):
    # with tf.variable_scope("optimizer"):
    lr_fn = None
    if params.train.learning_rate.type == "exponential":
        def exp_decay(epoch):
            initial_lrate = params.train.learning_rate.start_value
            k = params.train.learning_rate.end_value / params.train.learning_rate.start_value
            lrate = initial_lrate * math.exp(-k * epoch)
            return lrate
        lr_fn = exp_decay
        tf.summary.scalar('hyper_params/lr/start_value',
                          tf.constant(params.train.learning_rate.start_value))
        tf.summary.scalar('hyper_params/lr/end_value', tf.constant(params.train.learning_rate.end_value))
    elif params.train.learning_rate.type == "const":
        def const_lr(epoch):
            initial_lrate = params.train.learning_rate.start_value
            return initial_lrate
        lr_fn = const_lr
        tf.summary.scalar('hyper_params/lr/start_value',
                          tf.constant(params.train.learning_rate.start_value))
    else:
        raise RuntimeError("Unknown learning rate: %s" % params.train.learning_rate.type)
    lr_scheduler = LearningRateScheduler(lr_fn)
    learning_rate = params.train.learning_rate.start_value

    # Setup Optimizer
    optimizer = None
    if params.train.optimizer.type == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    elif params.train.optimizer.type == "rmsprop":
        optimizer = tf.keras.optimizers.RMSProp(lr=learning_rate)
    elif params.train.optimizer.type == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta(lr=learning_rate)
    elif params.train.optimizer.type == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate)
    elif params.train.optimizer.type == "adam":
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    else:
        raise RuntimeError("Unknown optimizer: %s" % params.train.optimizer.type)

    tf.summary.scalar('hyper_params/lr/learning_rate', optimizer.lr)

    return optimizer, lr_scheduler
