# MIT License
# 
# Copyright (c) 2018 Michael Fuerst
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

import tensorflow as tf
import math
LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler


def create_optimizer(params):
    learning_rate = None
    global_step = tf.train.get_global_step()
    if params.train.learning_rate.type == "exponential":
        learning_rate = tf.train.exponential_decay(params.train.learning_rate.start_value, global_step,
                                                   params.train.steps,
                                                   params.train.learning_rate.end_value / params.train.learning_rate.start_value,
                                                   staircase=False, name="lr_decay")
        tf.summary.scalar('hyper_params/lr/start_value',
                          tf.constant(params.train.learning_rate.start_value))
        tf.summary.scalar('hyper_params/lr/end_value', tf.constant(params.train.learning_rate.end_value))
    elif params.train.learning_rate.type == "const":
        learning_rate = tf.constant(params.train.learning_rate.start_value, dtype=tf.float32)
        tf.summary.scalar('hyper_params/lr/start_value',
                          tf.constant(params.train.learning_rate.start_value))
    else:
        raise RuntimeError("Unknown learning rate: %s" % params.train.learning_rate.type)
    tf.summary.scalar('hyper_params/lr/learning_rate', learning_rate)

    # Setup Optimizer
    optimizer = None
    if params.train.optimizer.type == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif params.train.optimizer.type == "rmsprop":
        decay = params.train.optimizer.get("decay", 0.9)
        momentum = params.train.optimizer.get("momentum", 0.0)
        epsilon = params.train.optimizer.get("epsilon", 1e-10)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum,
                                             epsilon=epsilon)
    elif params.train.optimizer.type == "adadelta":
        rho = params.train.optimizer.get("rho", 0.95)
        epsilon = params.train.optimizer.get("epsilon", 1e-08)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=rho,
                                              epsilon=epsilon)
    elif params.train.optimizer.type == "adagrad":
        initial_accumulator_value = params.train.optimizer.get("initial_accumulator_value", 0.1)
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                             initial_accumulator_value=initial_accumulator_value)
    elif params.train.optimizer.type == "adam":
        beta1 = params.train.optimizer.get("beta1", 0.9)
        beta2 = params.train.optimizer.get("beta2", 0.999)
        epsilon = params.train.optimizer.get("epsilon", 1e-08)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                          epsilon=epsilon)
    else:
        raise RuntimeError("Unknown optimizer: %s" % params.train.optimizer.type)

    return optimizer, global_step


def create_keras_optimizer(params):
    with tf.variable_scope("optimizer"):
        lr_fn = None
        if params.train.learning_rate.type == "exponential":
            def exp_decay(epoch):
                initial_lrate = params.train.learning_rate.start_value
                k = params.train.learning_rate.end_value / params.train.learning_rate.start_value
                lrate = initial_lrate * math.exp(-k * epoch)
                print(lrate)
                return lrate
            lr_fn = exp_decay
            tf.summary.scalar('hyper_params/lr/start_value',
                              tf.constant(params.train.learning_rate.start_value))
            tf.summary.scalar('hyper_params/lr/end_value', tf.constant(params.train.learning_rate.end_value))
        elif params.train.learning_rate.type == "const":
            def const_lr(epoch):
                initial_lrate = params.train.learning_rate.start_value
                print(lrate)
                return initial_lrate
            lr_fn = const_lr
            tf.summary.scalar('hyper_params/lr/start_value',
                              tf.constant(params.train.learning_rate.start_value))
        else:
            raise RuntimeError("Unknown learning rate: %s" % params.train.learning_rate.type)
        lr_sheduler = LearningRateScheduler(lr_fn)
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

        return optimizer, lr_sheduler
