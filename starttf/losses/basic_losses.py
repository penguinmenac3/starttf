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
