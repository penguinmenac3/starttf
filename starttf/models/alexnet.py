import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.contrib.framework.python.ops import arg_scope


trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def create_model(input_tensor, mode, hyper_params):
    """
    An alexnet network.

    :param input_tensor: The input tensor dict containing a "image" rgb tensor.
    :param mode: Execution mode as a tf.estimator.ModeKeys
    :param hyper_params: The hyper param file.
    :return: A dictionary containing all output tensors.
    """
    model = {}

    spatial_squeeze = False
    dropout_keep_prob = 0.5
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    num_classes = 10

    with tf.variable_scope('alexnet_v2') as scope:
        net = layers.conv2d(input_tensor, 64, [11, 11], 4, padding='VALID', scope='conv1')
        model["conv1"] = net
        net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
        model["pool1"] = net
        net = layers.conv2d(net, 192, [5, 5], scope='conv2')
        model["conv2"] = net
        net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
        model["pool2"] = net
        net = layers.conv2d(net, 384, [3, 3], scope='conv3')
        model["conv3"] = net
        net = layers.conv2d(net, 384, [3, 3], scope='conv4')
        model["conv4"] = net
        net = layers.conv2d(net, 256, [3, 3], scope='conv5')
        model["conv5"] = net
        net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')
        model["pool5"] = net

        # Use conv2d instead of fully_connected layers.
        with arg_scope(
              [layers.conv2d],
              weights_initializer=trunc_normal(0.005),
              biases_initializer=init_ops.constant_initializer(0.1)):
            net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
            model["fc6"] = net
            net = layers_lib.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
            model["fc7"] = net
            net = layers_lib.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = layers.conv2d(
                net,
                num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                biases_initializer=init_ops.zeros_initializer(),
                scope='fc8')

        # Convert end_points_collection into a end_point dict.
        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
            model['fc8'] = net

        # Collect outputs for api of network.
        model["logits"] = net
        model["probs"] = tf.nn.softmax(net)
    return model
