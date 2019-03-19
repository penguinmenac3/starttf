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

hyperparams = None


class RunOnce(object):
    def __init__(self, f):
        self.f = f
        self.called = False
    
    def __call__(self, *args, **kwargs):
        if not self.called:
            self.called = True
            self.f(*args, **kwargs)


class Module(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.hyperparams = hyperparams

    def __call__(self, *args, **kwargs):
        self.hyperparams = hyperparams

        with tf.variable_scope(self.name) as scope:
            return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """
        Run the model.
        """
        raise NotImplementedError("The model must implement a call function which predicts " +
                                  "the outputs (dict of tensors) given the input (dict of tensors).")
