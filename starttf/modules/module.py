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

import tensorflow as tf
from tensorflow.keras.layers import Lambda
import starttf


class Model(tf.keras.Model):
    def __init__(self, name, **kwargs):
        if starttf.modules.log_calls:
            print("{}.__init__".format(name))
        super(Model, self).__init__(name=name)
        self.__dict__.update(kwargs)

    def __call__(self, *args, **kwargs):
        if starttf.modules.log_calls or starttf.modules.log_inputs:
            print("{}.__call__".format(self.name))
        if starttf.modules.log_inputs:
            for i, val in enumerate(args):
                print(" {}: {}".format(i, val))
            for k in kwargs:
                print(" {}: {}".format(k, kwargs[k]))
            print()
        if len(args) > 0:
            return super(Model, self).__call__(*args, **kwargs)
        else:
            if "training" in kwargs:
                training = kwargs["training"]
                del kwargs["training"]
                return super(Model, self).__call__(kwargs, training=training)
            else:
                return super(Model, self).__call__(kwargs)

    def call(self, kwargs, training=False):
        return self.forward(training=training, **kwargs)

    def forward(self, training=False, **kwargs):
        raise RuntimeError("This function must be overwritten by a subclass!")


class Layer(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        if starttf.modules.log_calls:
            print("{}.__init__".format(name))
        super(Layer, self).__init__(name=name)
        self.__dict__.update(kwargs)

    def __call__(self, *args, **kwargs):
        if starttf.modules.log_calls or starttf.modules.log_inputs:
            print("{}.__call__".format(self.name))
        if starttf.modules.log_inputs:
            for i, val in enumerate(args):
                print(" {}: {}".format(i, val))
            for k in kwargs:
                print(" {}: {}".format(k, kwargs[k]))
            print()
        if len(args) > 0:
            return super(Layer, self).__call__(*args, **kwargs)
        else:
            if "training" in kwargs:
                training = kwargs["training"]
                del kwargs["training"]
                return super(Layer, self).__call__(kwargs, training=training)
            else:
                return super(Layer, self).__call__(kwargs)

    def call(self, kwargs, training=False):
        return self.forward(training=training, **kwargs)

    def forward(self, training=False, **kwargs):
        raise RuntimeError("This function must be overwritten by a subclass!")
