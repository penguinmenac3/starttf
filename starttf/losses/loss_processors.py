import tensorflow as tf


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
    # Broadcast multiply labels with alpha weights to select weights and then reduce them along last axis.
    weights = tf.reduce_sum(labels * alpha_weights, axis=-1)
    return weights * loss


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
    # Compute p_t that is used in paper.
    p_t = tf.reduce_sum(probs * labels, axis=-1) + tf.reduce_sum((1.0 - probs) * (1.0 - labels), axis=-1)
    return tf.pow(1.0 - p_t, gamma) * loss


def mask_loss(input_tensor, binary_tensor):
    """
    Mask a loss by using a tensor filled with 0 or 1.

    :param input_tensor: A float tensor of shape [batch_size, ...] representing the loss/cross_entropy
    :param binary_tensor: A float tensor of shape [batch_size, ...] representing the mask.
    :return: A float tensor of shape [batch_size, ...] representing the masked loss.
    """
    mask = tf.cast(tf.cast(binary_tensor, tf.bool), tf.float32)

    return input_tensor * mask
