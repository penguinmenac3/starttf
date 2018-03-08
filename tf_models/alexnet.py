import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.contrib.framework.python.ops import arg_scope


trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def create_model(hyper_params, input_tensor, reuse_weights=False, deploy_model=False, feed_dict={}):
    model = {}
    l2_weight = 0.0

    spatial_squeeze = False
    dropout_keep_prob = 0.5
    is_training = not deploy_model
    num_classes = 10

    with tf.variable_scope('alexnet_v2') as scope:
        if reuse_weights:
            scope.reuse_variables()

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
    return model, feed_dict


def create_loss(hyper_params, train_model, validation_model, train_labels, validation_labels=None):
    reports = []
    train_labels = tf.reshape(train_labels, [-1, 10])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_model["logits"], labels=train_labels))
    train_op = tf.train.RMSPropOptimizer(learning_rate=hyper_params.train.learning_rate,
                                         decay=hyper_params.train.decay).minimize(loss_op)
    tf.summary.scalar('train/loss', loss_op)
    reports.append(loss_op)

    # Create a validation loss if possible.
    if validation_labels is not None:
        validation_labels = tf.reshape(validation_labels, [-1, 10])
        validation_loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=validation_model["logits"], labels=validation_labels))
        tf.summary.scalar('validation/loss', validation_loss_op)
        reports.append(validation_loss_op)

    return train_op, reports
