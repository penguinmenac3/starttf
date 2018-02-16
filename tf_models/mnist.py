import tensorflow as tf


def create_model(hyper_params, input_tensor, reuse_weights=False, deploy_model=False, feed_dict={}):
    model = {}
    l2_weight = 0.0
    with tf.variable_scope('MnistNetwork') as scope:
        if reuse_weights:
            scope.reuse_variables()

        # Prepare the inputs
        x = tf.reshape(tensor=input_tensor, shape=(-1, 28, 28, 1), name="input")

        # First Conv Block
        conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(3, 3), strides=(1, 1), name="conv1",
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_weight), activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=(3, 3), strides=(1, 1), name="conv2",
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_weight), activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), name="pool2")

        # Second Conv Block
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=(3, 3), strides=(1, 1), name="conv3",
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_weight), activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=(3, 3), strides=(1, 1), name="conv4",
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_weight), activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(2, 2), strides=(2, 2), name="pool4")
        if not deploy_model:
            pool4 = tf.layers.dropout(inputs=pool4, rate=hyper_params.arch.dropout_rate, name="drop4")

        # Fully Connected Block
        probs = tf.layers.flatten(inputs=pool4)
        logits = tf.layers.dense(inputs=probs, units=10, activation=None, name="logits")
        probs = tf.nn.softmax(logits=logits, name="probs")

        # Collect outputs for api of network.
        model["pool2"] = pool2
        model["pool4"] = pool4
        model["logits"] = logits
        model["probs"] = probs
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
