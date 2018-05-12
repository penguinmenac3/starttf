import tensorflow as tf


def create_loss(model, labels, mode, hyper_params):
    """
    Create a cross entropy loss with the loss as the only metric.

    :param model: A dictionary containing all output tensors of your model.
    :param labels: A dictionary containing all label tensors.
    :param mode: tf.estimators.ModeKeys defining if you are in eval or training mode.
    :param hyper_params: A hyper parameters object.
    :return: A loss operation (tensor) and all the metrics(tensor dict) that should be logged for debugging.
    """
    metrics = {}
    losses = {}

    # Add loss
    labels = tf.reshape(labels["probs"], [-1, hyper_params.problem.number_of_categories])
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model["logits"], labels=labels)
    loss_op = tf.reduce_mean(ce)

    # Add losses to dict. "loss" is the primary loss that is optimized.
    losses["loss"] = loss_op
    metrics['accuracy'] = tf.metrics.accuracy(labels=labels,
                                              predictions=model["probs"],
                                              name='acc_op')

    return losses, metrics
