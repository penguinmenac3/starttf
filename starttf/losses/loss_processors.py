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
from starttf.utils.misc import tf_if


def interpolate_loss(labels, loss1, loss2, interpolation_values):
    """
    Interpolate two losses linearly.

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param loss1: A float tensor of shape [batch_size, ...] representing the loss1 for interpolation.
    :param loss2: A float tensor of shape [batch_size, ...] representing the loss2 for interpolation.
    :param interpolation_values: The values for each class how much focal loss should be interpolated in.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("interpolate_focus_loss"):
        # Select the probs or weights with the labels.
        t = tf.reduce_sum(labels * interpolation_values, axis=-1)
        return (1 - t) * loss1 + t * loss2


def alpha_balance_loss(labels, loss, alpha_weights):
    """
    Calculate the alpha balanced cross_entropy.

    This means for each sample the cross entropy is calculated and then weighted by the class specific weight.

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param loss: A float tensor of shape [batch_size, ...] representing the loss that should be focused.
    :param alpha_weights: A float tensor of shape [1, ..., num_classes] (... is filled with ones to match number
                              of dimensions to labels tensor) representing the weights for each class.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("alpha_balance"):
        # Broadcast multiply labels with alpha weights to select weights and then reduce them along last axis.
        weights = tf.reduce_sum(labels * alpha_weights, axis=-1)
        return weights * loss


def batch_alpha_balance_loss(labels, loss):
    """
    Calculate the alpha balanced cross_entropy.

    This means for each sample the cross entropy is calculated and then weighted by the class specific weight.

    There is yet no paper for this type of loss.

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param loss: A float tensor of shape [batch_size, ...] representing the loss that should be focused.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("batch_alpha_balance"):
        # Compute the occurrence probability for each class
        mu, _ = tf.nn.moments(labels, [0, 1, 2])

        # For weighting a class should be down weighted by its occurrence probability.
        not_mu = 1 - mu

        # Select the class specific not_mu
        not_mu_class = tf.reduce_sum(labels * not_mu, axis=-1)
        return not_mu_class * loss


def mask_loss(input_tensor, binary_tensor):
    """
    Mask a loss by using a tensor filled with 0 or 1.

    :param input_tensor: A float tensor of shape [batch_size, ...] representing the loss/cross_entropy
    :param binary_tensor: A float tensor of shape [batch_size, ...] representing the mask.
    :return: A float tensor of shape [batch_size, ...] representing the masked loss.
    """
    with tf.variable_scope("mask_loss"):
        mask = tf.cast(tf.cast(binary_tensor, tf.bool), tf.float32)

        return input_tensor * mask


def mean_on_masked(loss, mask, epsilon=1e-8, axis=None):
    """
    Average a loss correctly when it was masked.

    :param loss: A float tensor of shape [batch_size, ...] representing the (already masked) loss to be averaged.
    :param mask: A float tensor of shape [batch_size, ...] representing the mask.
    :param epsilon: Offset of log for numerical stability.
    :param axis: The dimensions to reduce. If None (the default), reduces all dimensions.
                 Must be in the range [-rank(input_tensor), rank(input_tensor)).
    """
    mask = tf.cast(tf.cast(mask, tf.bool), tf.float32)
    active_pixels = tf.reduce_sum(mask)
    active_pixels = tf_if(tf.equal(active_pixels, 0), epsilon, active_pixels)
    return tf.reduce_sum(loss, axis=axis) / active_pixels


def mask_and_mean_loss(input_tensor, binary_tensor, axis=None):
    """
    Mask a loss by using a tensor filled with 0 or 1 and average correctly.

    :param input_tensor: A float tensor of shape [batch_size, ...] representing the loss/cross_entropy
    :param binary_tensor: A float tensor of shape [batch_size, ...] representing the mask.
    :return: A float tensor of shape [batch_size, ...] representing the masked loss.
    :param axis: The dimensions to reduce. If None (the default), reduces all dimensions.
                 Must be in the range [-rank(input_tensor), rank(input_tensor)).
    """
    return mean_on_masked(mask_loss(input_tensor, binary_tensor), binary_tensor, axis=axis)


def variance_corrected_loss(loss, sigma_2=None):
    """
    Create a variance corrected loss.

    When summing variance corrected losses you get the same as multiloss.
    This is especially usefull for keras where when having multiple losses they are summed by keras.
    This multi-loss implementation is inspired by the Paper "Multi-Task Learning Using Uncertainty to Weight Losses
    for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.
    :param loss: The loss that should be variance corrected.
    :param sigma_2: Optional a variance (sigma squared) to use. If none is provided it is learned.
    :return: The variance corrected loss.
    """
    with tf.variable_scope("variance_corrected_loss"):
        sigma_cost = 0
        if sigma_2 is None:
            # FIXME the paper has been updated Apr 2018, check if implementation is still valid.
            sigma = tf.get_variable(name="sigma", dtype=tf.float32, initializer=tf.constant(1.0), trainable=True)
            sigma_2 = tf.pow(sigma, 2)
            tf.summary.scalar("sigma2", sigma_2)
            sigma_cost = tf.log(sigma_2 + 1.0)
        return 0.5 / sigma_2 * loss + sigma_cost


def multiloss(losses, logging_namespace="multiloss", exclude_from_weighting=[]):
    """
    Create a loss from multiple losses my mixing them.
    This multi-loss implementation is inspired by the Paper "Multi-Task Learning Using Uncertainty to Weight Losses
    for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.
    :param losses: A dict containing all losses that should be merged.
    :param logging_namespace: Variable scope in which multiloss lives.
    :param exclude_from_weighting: A list of losses that are already weighted and should not be sigma weighted.
    :return: A single loss.
    """
    with tf.variable_scope(logging_namespace):
        sum_loss = 0
        for loss_name, loss in losses.items():
            if loss_name not in exclude_from_weighting:
                with tf.variable_scope(loss_name) as scope:
                    sum_loss += variance_corrected_loss(loss)
            else:
                sum_loss += loss
        return sum_loss


def focus_loss(labels, probs, loss, gamma):
    """
    Calculate the alpha balanced focal loss.

    See the focal loss paper: "Focal Loss for Dense Object Detection" [by Facebook AI Research]

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param probs: A float tensor of shape [batch_size, ..., num_classes] representing the probs (after softmax).
    :param loss: A float tensor of shape [batch_size, ...] representing the loss that should be focused.
    :param gamma: The focus parameter.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("focus_loss"):
        # Compute p_t that is used in paper.
        # FIXME is it possible that the 1-p term does not make any sense?
        p_t = tf.reduce_sum(probs * labels, axis=-1)# + tf.reduce_sum((1.0 - probs) * (1.0 - labels), axis=-1)

        focal_factor = tf.pow(1.0 - p_t, gamma) if gamma > 0 else 1  # Improve stability for gamma = 0
        return tf.stop_gradient(focal_factor) * loss
