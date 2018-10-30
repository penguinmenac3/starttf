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
import keras


class StartTFPartialModel(object):
    def __init__(self, hyperparams):
        tf.keras.backend.set_session(tf.get_default_session())
        self.hyperparams = hyperparams
        self.tensorflow = False
        self.keras = False

    def __call__(self, input_tensor, training=False, for_tensorflow=False, for_keras=False):
        if for_tensorflow:
            self.tensorflow = True
        if for_keras:
            self.keras = True
        if not self.tensorflow and not self.keras:
            raise RuntimeError("Your model has no specification if it is for tensorflow or keras, pass the optional arguments.")
        if self.tensorflow and self.keras:
            raise RuntimeError("Your model cannot be for keras and tensorflow at the same time. You have to choose one.")
        return self.call(input_tensor, training)

    def call(self, input_tensor, training=False):
        """
        Run the model.
        """
        raise NotImplementedError("The model must implement a call function which predicts " +
                                  "the outputs (dict of tensors) given the input (dict of tensors).")
        return {}, {}


class StartTFFromKerasBackbone(StartTFPartialModel):
    def __init__(self, hyperparams, model, outputs):
        """
        Extract the backbone of a model

        :param hyperparams: The hyperparameters for the model.
        :param model: A keras model.
        :param outputs: A List[str] of layer names, as shown in model.summary().
        """
        super(StartTFFromKerasBackbone, self).__init__(hyperparams)
        assert outputs is not None
        self.outputs = outputs
        layers = [model.get_layer(x).output for x in outputs]
        # Using keras and tf.keras seems to make a difference here. tf.keras does not work.
        self.model = keras.models.Model(inputs=model.inputs,
                                        outputs=layers)

    def call(self, input_tensor, training=False):
        model = {}
        debug = {}
        with tf.variable_scope("backbone"):
            # Get Input
            image = input_tensor["image"]
            if self.tensorflow:
                image = tf.cast(image, dtype=tf.float32, name="input/cast")

            # Create the model
            backbone = self.model(image)

            # Repack as dict for partial model
            if self.keras:
                model = {name: backbone.get_layer(name).output for name in self.outputs}
            if self.tensorflow:
                tmp = dict(zip(self.outputs, backbone))
                model = {k: tf.identity(feature) for k, feature in tmp.iteritems()}
            debug["image"] = image
        return model, debug


class StartTFModel(StartTFPartialModel):
    def __init__(self, hyperparams):
        super(StartTFModel, self).__init__(hyperparams)

    def create_tf_model(self, input_tensor, training=False):
        with tf.variable_scope("model"):
            output_tensors, debug_tensors = self.__call__(input_tensor, training, for_tensorflow=True, for_keras=False)
            output_tensors.update(debug_tensors)
            return output_tensors

    def create_keras_model(self, input_tensor, training=False):
        with tf.variable_scope("model"):
            output_tensors, debug_tensors = self.__call__(input_tensor, training, for_tensorflow=False, for_keras=True)
            output_tensors.update(debug_tensors)

            input_tensor_order = sorted(list(input_tensor.keys()))
            inputs = [input_tensor[k] for k in input_tensor_order]
            outputs = [tf.keras.layers.Lambda(lambda x: x, name=k)(output_tensors[k]) for k in output_tensors]

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model


class RLModel(StartTFModel):
    def __init__(self, hyperparams):
        super(RLModel, self).__init__(hyperparams)
        self.model = None

    def update(self, **kwargs):
        """
        Train the reinforcement learning model on an example.

        The exact structure and parameters of the training are open.
        """
        # TODO is there something that can be done to help with tf eager learning?
        # I guess in RL eager makes sense.
        raise NotImplementedError()
