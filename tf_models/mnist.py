import tensorflow as tf

def create_model(input_tensor, mode, hyper_params):
    model = {}
    l2_weight = 0.0
    with tf.variable_scope('MnistNetwork') as scope:
        if mode == tf.estimator.ModeKeys.EVAL:
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
        if mode == tf.estimator.ModeKeys.TRAIN:
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
    return model
