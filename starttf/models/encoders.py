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

from starttf.models.model import StartTFPartialModel

ENCODERS = {
    "vgg16": tf.keras.applications.vgg16.VGG16,
    "vgg19": tf.keras.applications.vgg19.VGG19,
    "xception": tf.keras.applications.xception.Xception,
    "resnet50": tf.keras.applications.resnet50.ResNet50,
    "inception_resnet_v2": tf.keras.applications.inception_resnet_v2.InceptionResNetV2
}


class Encoder(StartTFPartialModel):
    def __init__(self, hyperparams):
        super(Encoder, self).__init__(hyperparams)
        encoder_name = hyperparams.get("encoder", "vgg16")
        if encoder_name not in ENCODERS:
            errormsg = "Unknown encoder {}. Please pick one of the following: {}".format(encoder_name, ENCODERS.keys())
            raise ValueError(errormsg)

        encoder_weights = hyperparams.get("encoder_weights", "imagenet")
        self.encoder = ENCODERS[encoder_name](weights=encoder_weights, include_top=False)

    def call(self, input_tensor, training=False):
        """
        Run the model.
        """
        image = tf.cast(input_tensor["image"], dtype=tf.float32, name="input/cast")
        model = {}
        debug = {}
        debug["image"] = image
        model["features"] = self.encoder(image, training=training)
        return model, debug


class MultiResolutionEncoder(StartTFPartialModel):
    def __init__(self, hyperparams):
        super(Encoder, self).__init__(hyperparams)
        encoder_name = hyperparams.get("encoder", "vgg16")
        if encoder_name not in ENCODERS:
            errormsg = "Unknown encoder {}. Please pick one of the following: {}".format(encoder_name, ENCODERS.keys())
            raise ValueError(errormsg)

        encoder_weights = hyperparams.get("encoder_weights", "imagenet")
        self.encoder = ENCODERS[encoder_name](weights=encoder_weights, include_top=False)
        if encoder_name == "resnet50":
            # FPN Paper style resnet50 encoder
            C3 = resnet50.get_layer("activation_22").output
            C4 = resnet50.get_layer("activation_40").output
            C5 = resnet50.get_layer("activation_49").output
            self.encoder = tf.keras.Model(inputs=self.encoder.inputs, outputs=[C3, C4, C5])
        elif encoder_name == "vgg16":
            # TODO implement for SSD style multires vgg encoder
            raise NotImplementedError("Not yet implemented.")
        else:
            raise RuntimeError("Other encoder networks are not supported.")

    def call(self, input_tensor, training=False):
        """
        Run the model.
        """
        image = tf.cast(input_tensor["image"], dtype=tf.float32, name="input/cast")
        model = {}
        debug = {}
        debug["image"] = image
        model["features"] = self.encoder(image, training=training)
        return model, debug
