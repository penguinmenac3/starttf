import tensorflow as tf


def create_model(input_tensor, mode, hyper_params):
    """
    A wrapper for arbitrary image processing models from the tf hub.

    :param input_tensor: The input tensor dict containing a "image" rgb tensor.
    :param mode: Execution mode as a tf.estimator.ModeKeys
    :param hyper_params: The hyper param file. {"tf_hub_wrapper": {"model_url": "https://tfhub.dev/...", "trainable": false}}
    :return: A dictionary containing all output tensors.
    """
    model = {}
    with tf.variable_scope('tf_hub_wrapper') as scope:
        module = tf.contrib.hub.Module(hyper_params.tf_hub_wrapper.model_url, trainable=hyper_params.tf_hub_wrapper.trainable)
        model["output"] = module(input_tensor["image"] / 255.0)
    return model
