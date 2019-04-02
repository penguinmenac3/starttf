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
import starttf as m

Lambda = tf.keras.layers.Lambda


class Module(object):
    """
    By subclassing `Module` instead of `object` any `tf.Variable` or
    `Module` instances assigned to object properties can be collected using
    the `variables`, `trainable_variables` or `submodules` property.

    You must implement the call method to use the variables.
    For complex variables might using a build function with @Once annotation can be usefull.
    """
    def __init__(self, name, **kwargs):
        self.__name = name
        if m.hyperparams is None and m.hyperparams is not m.NO_PARAMS:
            raise RuntimeWarning("You did not set starttf.modules.hyperparams. You may want to consider setting it to starttf.modules.NO_PARAMS if this was intentional")
        self.hyperparams = m.hyperparams
        self.lambda_mode = False
        self.__dict__.update(kwargs)
        if self.lambda_mode:
            self.__lambda = Lambda(self.call)

    @property
    def name(self):
        return self.__name

    @property
    def submodules(self):
        modules = []
        for k in self.__dict__:
            v = self.__dict__[k]
            if isinstance(v, Module):
                modules.append(v)
                modules.append(v.submodules)
        return modules

    @property
    def variables(self):
        all_vars = []
        for k in self.__dict__:
            v = self.__dict__[k]
            if isinstance(v, Module):
                all_vars.append(v.variables)
            if isinstance(v, tf.Variable):
                all_vars.append(v)
        return all_vars

    @property
    def trainable_variables(self):
        all_vars = self.variables
        train_vars = []
        for v in all_vars:
            if v.trainable:
                train_vars.append(v)
        return train_vars

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.name) as scope:    
            if self.lambda_mode:
                return self.__lambda(*args, **kwargs)
            return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """
        Run the model.
        """
        raise NotImplementedError("The model must implement a call function which predicts " +
                                  "the outputs (dict of tensors) given the input (dict of tensors).")

    def create_tf_model(self, **kwargs):
        self.tensorflow = True
        self.keras = False
        with tf.variable_scope("model"):
            output_tensors = self.__call__(**kwargs)
            if isinstance(output_tensors, tuple):
                output_tensors = output_tensors.__dict__
            return output_tensors

    def create_keras_model(self, **kwargs):
        self.tensorflow = False
        self.keras = True
        with tf.variable_scope("model"):
            output_tensors = self.__call__(**kwargs)
            if isinstance(output_tensors, tuple):
                output_tensors = output_tensors.__dict__
            input_tensor = {}
            for k in kwargs:
                if tf.contrib.framework.is_tensor(kwargs[k]):
                    input_tensor[k] = kwargs[k]

            input_tensor_order = sorted(list(input_tensor.keys()))
            inputs = [input_tensor[k] for k in input_tensor_order]
            outputs = [tf.keras.layers.Lambda(lambda x: x, name=k)(output_tensors[k]) for k in output_tensors]

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model
