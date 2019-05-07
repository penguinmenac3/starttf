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

import os
import tensorflow as tf
import starttf
from starttf import RunOnce

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
        if starttf.modules.log_calls:
            print("{}.__init__".format(name))
        self.__name = name
        self.__outputs = None
        self.__model = None
        if starttf.hyperparams is None and starttf.hyperparams is not starttf.NO_PARAMS:
            raise RuntimeWarning("You did not set starttf.modules.hyperparams. You may want to consider setting it to starttf.modules.NO_PARAMS if this was intentional")
        self.hyperparams = starttf.hyperparams
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
        if starttf.modules.log_calls or starttf.modules.log_inputs:
            print("{}.__call__".format(self.name))
        if starttf.modules.log_inputs:
            for a in args:
                print(" {}".format(a))
            for k in kwargs:
                print(" {}: {}".format(k, kwargs[k]))
            print()
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:    
            if self.lambda_mode:
                return self.__lambda(*args, **kwargs)
            return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """
        Run the model.
        """
        raise NotImplementedError("The model must implement a call function which predicts " +
                                  "the outputs (dict of tensors) given the input (dict of tensors).")

    @RunOnce
    def summary(self, **kwargs):
        for k in kwargs:
            tf.summary.scalar(k, kwargs[k])

    def create_tf_model(self, **kwargs):
        self.tensorflow = True
        self.keras = False
        with tf.variable_scope("model"):
            output_tensors = self.__call__(**kwargs)
            if isinstance(output_tensors, tuple):
                output_tensors = output_tensors.__dict__
            self.__model = output_tensors
            self.__outputs = list(output_tensors.keys())
            return self.__model

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
            self.__outputs = list(output_tensors.keys())
            outputs = [tf.keras.layers.Lambda(lambda x: x, name=k)(output_tensors[k]) for k in self.__outputs]

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.__model = model
            return self.__model

    def load_model(self, checkpoint_path, input_shapes_dict, input_dtypes_dict, model_path=None):
        """
        Load a model from a checkpoint path or set checkpoint path to None and provide a model_path to load a model saved with save_model.
        """
        if model_path is None:
            model_path = os.path.join(checkpoint_path, "model.hdf5")
            if not os.path.exists(model_path):
                # If it is not a keras model it is a tensorflow checkpoint
                model_path = checkpoint_path
        if (model_path.endswith(".h5") or model_path.endswith(".hdf5")) and os.path.exists(model_path):
            input_dict = {k: tf.keras.layers.Input(dtype=input_dtypes_dict[k], shape=input_shapes_dict[k], name="{}".format(k)) for k in input_shapes_dict.keys()}
            self.__input_dict = input_dict
            self.__model = self.create_keras_model(**input_dict)
            self.__model.load_weights(model_path)
        elif os.path.exists(model_path):
            saver = tf.train.Saver()
            input_dict = {k: tf.placeholder(dtype=input_dtypes_dict[k], shape=input_shapes_dict[k], name="{}".format(k)) for k in input_shapes_dict.keys()}
            self.__input_dict = input_dict
            self.__model = self.create_tf_model(**input_dict)
            saver.restore(tf.get_default_session(), model_path)
        else:
            raise RuntimeError("The specified model does not exist!")

    def save_model(self, model_path):
        """
        Save the model in a format that can be loaded via load_model.
        """
        if self.keras and self.__model is not None:
            if not model_path.endswith(".h5") and not model_path.endswith(".hdf5"):
                raise RuntimeError("model_path must end on .h5 or .hdf5 for keras models!")
            self.__model.save_weights(model_path)
        elif self.tensorflow and self.__model is not None:
            if not model_path.endswith(".ckpt"):
                raise RuntimeError("model_path must end on .ckpt for tensorflow models!")
            saver = tf.train.Saver()
            save_path = saver.save(tf.get_default_session(), model_path)
        else:
            raise RuntimeError("You must first create a tensorflow or keras model directly or load it via the load_model function.")

    def predict(self, inputs, input_placeholders=None):
        """
        Predict the outputs of a model given some inputs.
        """
        if self.keras and self.__model is not None:
            outputs = self.__model.predict(inputs)
            outputs = dict(zip(self.__outputs, outputs))
            return outputs
            # Unpack since it gives a batch of 1 output
            #if isinstance(outputs, dict):
            #    outputs = {k: outputs[k][0] for k in outputs.keys()}
            #else:
            #    outputs = outputs[0]
            #return outputs
        elif self.tensorflow and self.__model is not None:
            sess = tf.get_default_session()
            if input_placeholders is None:
                input_placeholders = self.__input_dict
            if input_placeholders is None:
                raise RuntimeError("You must either provide your input_placeholders or create the model via load_model(...)!")
            feed_dict = {input_placeholders[k]: inputs[k] for k in inputs.keys()}
            outputs = sess.run(self.__model, feed_dict=feed_dict)
            # TODO is input batch packing or output batch unpacking required?
            return outputs
        else:
            raise RuntimeError("You must first create a tensorflow or keras model directly or load it via the load_model function.")
