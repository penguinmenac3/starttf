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


def tile_2d(input, k_x, k_y, name, reorder_required=True):
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
    size = input.get_shape().as_list()
    c, h, w = size[3], size[1], size[2]
    batch_size = size[0]
    if batch_size is None:
        batch_size = -1

    # Check if tiling is possible and define output shape.
    assert c % (k_x * k_y) == 0

    tmp = input

    if reorder_required:
        output_channels = int(c / (k_x * k_y))
        channels = tf.unstack(tmp, axis=-1)
        reordered_channels = [None for _ in range(len(channels))]
        for o in range(output_channels):
            for i in range(k_x * k_y):
                target = o + i * output_channels
                source = o * (k_x * k_y) + i
                reordered_channels[target] = channels[source]
        tmp = tf.stack(reordered_channels, axis=-1)

    # Actual tilining
    with tf.variable_scope(name) as scope:
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, w, int(h * k_y), int(c / (k_y))))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, int(h * k_y), int(w * k_x), int(c / (k_y * k_x))))
    
    return tmp


def inverse_tile_2d(input, k_x, k_y, name):
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
    assert w % k_x == 0 and h % k_y == 0

    # Actual inverse tilining
    with tf.variable_scope(name) as scope:
        tmp = input
        tmp = tf.reshape(tmp, (batch_size, int(h * k_y), w, int(c * k_x)))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, w, h, int(c * k_y * k_x)))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])

    return tmp


def feature_passthrough(early_feat, late_feat, filters, name, kernel_size=(1, 1)):
    """
    A feature passthrough layer inspired by yolo9000 and the inverse tiling layer.

    It can be proven, that this layer does the same as conv(concat(inverse_tile(early_feat), late_feat)).
    This layer has no activation function.

    :param early_feat: The early feature layer of shape [batch_size, h * s_x, w * s_y, _].
    s_x and s_y are integers computed internally describing the scale between the layers.
    :param late_feat:  The late feature layer of shape [batch_size, h, w, _].
    :param filters: The number of convolution filters.
    :param name: The name of the layer.
    :param kernel_size: The size of the kernel. Default (1x1).
    :return: The output tensor of shape [batch_size, h, w, outputs]
    """
    _, h_early, w_early, c_early = early_feat.get_shape().as_list()
    _, h_late, w_late, c_late = late_feat.get_shape().as_list()

    s_x = int(w_early / w_late)
    s_y = int(h_early / h_late)

    assert h_late * s_y == h_early and w_late * s_x == w_early

    with tf.variable_scope(name) as scope:
        early_conv = tf.layers.conv2d(early_feat, filters=filters, kernel_size=(s_x * kernel_size[0], s_y * kernel_size[1]), strides=(s_x, s_y), padding="same")
        late_conv = tf.layers.conv2d(late_feat, filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        return early_conv + late_conv


def upsampling_feature_passthrough(early_feat, late_feat, filters, name, kernel_size=(1, 1)):
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
    _, h_early, w_early, c_early = early_feat.get_shape().as_list()
    _, h_late, w_late, c_late = late_feat.get_shape().as_list()

    s_x = int(w_early / w_late)
    s_y = int(h_early / h_late)

    assert h_late * s_y == h_early and w_late * s_x == w_early

    with tf.variable_scope(name) as scope:
        tiled = tile_2d(late_feat, s_x, s_y, "tile_2d", reorder_required=False)
        concated = tf.concat([early_feat, tiled], axis=-1)
        return tf.layers.conv2d(concated, filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
