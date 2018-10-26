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


def sum_abs_distance(labels, preds):
    """
    Compute the sum of abs distances.

    :param labels: A float tensor of shape [batch_size, ..., X] representing the labels.
    :param preds: A float tensor of shape [batch_size, ..., X] representing the predictions.
    :return: A float tensor of shape [batch_size, ...] representing the summed absolute distance.
    """
    with tf.variable_scope("sum_abs_distance"):
        return tf.reduce_sum(tf.abs(preds - labels), axis=-1)


def l1_distance(labels, preds):
    """
    Compute the l1_distance.

    :param labels: A float tensor of shape [batch_size, ..., X] representing the labels.
    :param preds: A float tensor of shape [batch_size, ..., X] representing the predictions.
    :return: A float tensor of shape [batch_size, ...] representing the l1 distance.
    """
    with tf.variable_scope("l1_distance"):
        return tf.norm(preds - labels, ord=1)


def smooth_l1_distance(labels, preds, delta=1.0):
    """
    Compute the smooth l1_distance.

    :param labels: A float tensor of shape [batch_size, ..., X] representing the labels.
    :param preds: A float tensor of shape [batch_size, ..., X] representing the predictions.
    :param delta: `float`, the point where the huber loss function changes from a quadratic to linear.
    :return: A float tensor of shape [batch_size, ...] representing the smooth l1 distance.
    """
    with tf.variable_scope("smooth_l1"):
        return tf.reduce_sum(tf.losses.huber_loss(
            labels=labels,
            predictions=preds,
            delta=delta,
            loss_collection=None,
            reduction=tf.losses.Reduction.NONE
        ), axis=-1)


def l2_distance(labels, preds):
    """
    Compute the l2_distance.

    :param labels: A float tensor of shape [batch_size, ..., X] representing the labels.
    :param preds: A float tensor of shape [batch_size, ..., X] representing the predictions.
    :return: A float tensor of shape [batch_size, ...] representing the l2 distance.
    """
    with tf.variable_scope("l2_distance"):
        return tf.norm(preds - labels, ord=2)


def cross_entropy(labels, logits):
    """
    Calculate the cross_entropy.

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param logits: A float tensor of shape [batch_size, ..., num_classes] representing the logits.
    :return: A tensor representing the cross entropy.
    """
    with tf.variable_scope("cross_entropy"):
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
