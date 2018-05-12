import tensorflow as tf


def _create_conv2_block(model, net, filters, layer_number):
    net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=(3, 3), strides=(1, 1), name="conv%d_1" % layer_number,
                           activation=tf.nn.relu, padding="same")
    model["vgg16/conv%d_1" % layer_number] = net
    net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=(3, 3), strides=(1, 1), name="conv%d_2" % layer_number,
                           activation=tf.nn.relu, padding="same")
    model["vgg16/conv%d_2" % layer_number] = net
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), padding="same", name="pool%d" % layer_number)
    model["vgg16/pool%d" % layer_number] = net
    return net


def _create_conv3_block(model, net, filters, layer_number):
    net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=(3, 3), strides=(1, 1), name="conv%d_1" % layer_number,
                           activation=tf.nn.relu, padding="same")
    model["vgg16/conv%d_1" % layer_number] = net
    net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=(3, 3), strides=(1, 1), name="conv%d_2" % layer_number,
                           activation=tf.nn.relu, padding="same")
    model["vgg16/conv%d_2" % layer_number] = net
    net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=(3, 3), strides=(1, 1), name="conv%d_3" % layer_number,
                           activation=tf.nn.relu, padding="same")
    model["vgg16/conv%d_3" % layer_number] = net
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), padding="same", name="pool%d" % layer_number)
    model["vgg16/pool%d" % layer_number] = net
    return net


def create_model(input_tensor, mode, hyper_params):
    """
    A full reference model of vgg16 without pretrained weights.

    This uses the layers api and is optimized for tensorflow.
    :param input_tensor: The input tensor dict containing a "image" rgb tensor.
    :param mode: Execution mode as a tf.estimator.ModeKeys
    :param hyper_params: The hyper param file. "vgg16" : {"encoder_only": Boolean}
    :return: A dictionary containing all output tensors.
    """
    model = {}
    with tf.variable_scope('vgg16') as scope:
        net = tf.cast(input_tensor["image"], dtype=tf.float32, name="input/cast")
        model["image"] = net
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net = net - mean
        model["image-normalized"] = net

        net = _create_conv2_block(model, net, filters=64, layer_number=1)
        net = _create_conv2_block(model, net, filters=128, layer_number=2)
        net = _create_conv3_block(model, net, filters=256, layer_number=3)
        net = _create_conv3_block(model, net, filters=512, layer_number=4)
        net = _create_conv3_block(model, net, filters=512, layer_number=5)
        print(net.get_shape())

        if not hyper_params.vgg16.encoder_only:
            net = tf.layers.conv2d(inputs=net, filters=4096, kernel_size=(1, 1), strides=(1, 1), name="fc1", activation=tf.nn.relu)
            model["vgg16/fc1"] = net
            net = tf.layers.conv2d(inputs=net, filters=4096, kernel_size=(1, 1), strides=(1, 1), name="fc2", activation=tf.nn.relu)
            model["vgg16/fc2"] = net
            net = tf.layers.conv2d(inputs=net, filters=1000, kernel_size=(1, 1), strides=(1, 1), name="logits", activation=None)
            model["logits"] = net
            net = tf.nn.softmax(net)
            model["probs"] = net
    return model
