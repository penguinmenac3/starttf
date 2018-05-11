import tensorflow as tf

from starttf.utils.misc import mode_to_str


def create_loss(model, labels, mode, hyper_params):
    """
    Create a cross entropy loss with the loss as the only metric.

    :param model: A dictionary containing all output tensors of your model.
    :param labels: A dictionary containing all label tensors.
    :param mode: tf.estimators.ModeKeys defining if you are in eval or training mode.
    :param hyper_params: A hyper parameters object.
    :return: A loss operation (tensor) and all the metrics(tensor dict) that should be logged for debugging.
    """
    mode_name = mode_to_str(mode)
    metrics = {}

    # Add loss
    labels = tf.reshape(labels["probs"], [-1, hyper_params.arch.output_dimension])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model["logits"], labels=labels))
    tf.summary.scalar(mode_name + '/loss', loss_op)
    metrics[mode_name + '/loss'] = loss_op

    return loss_op, metrics
