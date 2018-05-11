import tensorflow as tf


def create_model(input_tensor, mode, hyper_params):
    """
    A wrapper for inception v3 feature encoder from the tf hub.

    :param input_tensor: The input tensor dict containing a "image" rgb tensor.
    :param mode: Execution mode as a tf.estimator.ModeKeys
    :param hyper_params: The hyper param file. {"inception_v3": {"trainable": false}}
    :return: A dictionary containing all output tensors.
    """
    model = {}
    with tf.variable_scope('inception_v3') as scope:
        if mode == tf.estimator.ModeKeys.EVAL:
            scope.reuse_variables()

        module = tf.contrib.hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=hyper_params.inception_v3.trainable)
        model["features"] = module(input_tensor["image"] / 255.0)
    return model
