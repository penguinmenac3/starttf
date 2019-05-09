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

import tensorflow as tf
from starttf.modules import Module
from starttf import RunOnce

Conv2D = tf.keras.layers.Conv2D


class Tile2D(Module):
    def __init__(self, k_x, k_y, name="Tile2D"):
        super().__init__(name, lambda_mode=True)
        self.k_x = k_x
        self.k_y = k_y

    def call(self, input_tensor):
        """
        A tiling layer like introduced in overfeat and huval papers.
        :param input: Your input tensor.
        :param k_x: The tiling factor in x direction.
        :param k_y: The tiling factor in y direction.
        :param name: The name of the layer.
        :param reorder_required: To implement an exact huval tiling you need reordering.
        However not using it is more efficient and when training from scratch setting this to false is highly recommended.
        :return: The output tensor.
        """
        size = input_tensor.get_shape().as_list()
        c, h, w = size[3], size[1], size[2]
        batch_size = size[0]
        if batch_size is None:
            batch_size = -1

        # Check if tiling is possible and define output shape.
        assert c % (self.k_x * self.k_y) == 0

        tmp = input_tensor

        # Actual tilining
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, w, int(h * self.k_y), int(c / (self.k_y))))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, int(h * self.k_y), int(w * self.k_x), int(c / (self.k_y * self.k_x))))

        return tmp


class InverseTile2D(Module):
    def __init__(self, k_x, k_y, name="InverseTile2D"):
        super().__init__(name, lambda_mode=True)
        self.k_x = k_x
        self.k_y = k_y

    def call(self, input_tensor):
        """
            An inverse tiling layer.

            An inverse to the tiling layer can be of great use, since you can keep the resolution of your output low,
            but harness the benefits of the resolution of a higher level feature layer.
            If you insist on a source you can call it very lightly inspired by yolo9000 "passthrough layer".

            :param input: Your input tensor. (Assert input.shape[1] % k_y = 0 and input.shape[2] % k_x = 0)
            :param k_x: The tiling factor in x direction [int].
            :param k_y: The tiling factor in y direction [int].
            :param name: The name of the layer.
            :return: The output tensor of shape [batch_size, inp.height / k_y, inp.width / k_x, inp.channels * k_x * k_y].
            """

        batch_size, h, w, c = input.get_shape().as_list()
        if batch_size is None:
            batch_size = -1

        # Check if tiling is possible and define output shape.
        assert w % self.k_x == 0 and h % self.k_y == 0

        # Actual inverse tilining
        tmp = input
        tmp = tf.reshape(tmp, (batch_size, int(h * self.k_y), w, int(c * self.k_x)))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, w, h, int(c * self.k_y * self.k_x)))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])

        return tmp


class FeaturePassthrough(Module):
    def __init__(self, filters, kernel_size=(1, 1), name="FeaturePassthrough"):
        """
        Create an inverse tiling module.

        :param filters: The number of convolution filters.
        :param kernel_size: The size of the kernel. Default (1x1).
        :param name: The name of the layer.
        """
        super().__init__(name, lambda_mode=True)
        self.filters = filters
        self.kernel_size = kernel_size

    @RunOnce
    def build(self, early_feat, late_feat):
        _, h_early, w_early, c_early = early_feat.get_shape().as_list()
        _, h_late, w_late, c_late = late_feat.get_shape().as_list()

        s_x = int(w_early / w_late)
        s_y = int(h_early / h_late)

        assert h_late * s_y == h_early and w_late * s_x == w_early

        self.early_conv = Conv2D(filters=self.filters, kernel_size=(
            s_x * self.kernel_size[0], s_y * self.kernel_size[1]), strides=(s_x, s_y), padding="same")
        self.late_conv = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=(1, 1), padding="same")

    def call(self, early_feat, late_feat):
        """
        A feature passthrough layer inspired by yolo9000 and the inverse tiling layer.

        It can be proven, that this layer does the same as conv(concat(inverse_tile(early_feat), late_feat)).
        This layer has no activation function.

        :param early_feat: The early feature layer of shape [batch_size, h * s_x, w * s_y, _].
        s_x and s_y are integers computed internally describing the scale between the layers.
        :param late_feat:  The late feature layer of shape [batch_size, h, w, _].
        :return: The output tensor of shape [batch_size, h, w, outputs]
        """
        self.build(early_feat, late_feat)
        return self.early_conv(early_feat) + self.late_conv(late_feat)


class UpsamplingFeaturePassthrough(Module):
    def __init__(self, filters, kernel_size=(1, 1), name="UpsamplingFeaturePassthrough"):
        """
        Create an upsampling feature passthrough.

        :param filters: The number of convolution filters.
        :param kernel_size: The size of the kernel. Default (1x1).
        :param name: The name of the layer.
        """
        super().__init__(name, lambda_mode=True)
        self.filters = filters
        self.kernel_size = kernel_size

    @RunOnce
    def build(self, early_feat, late_feat):
        _, h_early, w_early, c_early = early_feat.get_shape().as_list()
        _, h_late, w_late, c_late = late_feat.get_shape().as_list()

        s_x = int(w_early / w_late)
        s_y = int(h_early / h_late)

        assert h_late * s_y == h_early and w_late * s_x == w_early

        self.tiling = Tile2D(s_x, s_y)
        self.conv = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=(1, 1), padding="same")

    def call(self, early_feat, late_feat):
        """
        An upsampling feature passthrough layer inspired by yolo9000 and the tiling layer.

        It can be proven, that this layer does the same as conv(concat(early_feat, tile_2d(late_feat))).
        This layer has no activation function.

        :param early_feat: The early feature layer of shape [batch_size, h * s_x, w * s_y, _].
        s_x and s_y are integers computed internally describing the scale between the layers.
        :param late_feat:  The late feature layer of shape [batch_size, h, w, _].
        :param filters: The number of convolution filters.
        :param name: The name of the layer.
        :param kernel_size: The size of the kernel. Default (1x1).
        :return: The output tensor of shape [batch_size, h * s_x, w * s_y, outputs]
        """
        self.build(early_feat, late_feat)
        tiled = self.tiling(late_feat)
        concated = tf.concat([early_feat, tiled], axis=-1)
        return self.conv(concated)
