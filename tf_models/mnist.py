import tensorflow as tf
from tf_models.model import Model


class Mnist(Model):
    def __init__(self, hyper_params_filepath):
        super(Mnist, self).__init__(hyper_params_filepath)

    def _create_model(self, input_tensor, reuse_weights, is_deploy_model=False):
        outputs = {}
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
            if not is_deploy_model:
                pool4 = tf.layers.dropout(inputs=pool4, rate=self.hyper_params.arch.dropout_rate, name="drop4")

            # Fully Connected Block
            probs = tf.layers.flatten(inputs=pool4)
            logits = tf.layers.dense(inputs=probs, units=10, activation=None, name="logits")
            probs = tf.nn.softmax(logits=logits, name="probs")

            # Collect outputs for api of network.
            outputs["pool2"] = pool2
            outputs["pool4"] = pool4
            outputs["logits"] = logits
            outputs["probs"] = probs
        return outputs

    def _create_loss(self, labels, validation_labels=None):
        labels = tf.reshape(labels, [-1, 10])
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model_train["logits"], labels=labels))
        train_op = tf.train.RMSPropOptimizer(learning_rate=self.hyper_params.train.learning_rate, decay=self.hyper_params.train.decay).minimize(loss_op)
        tf.summary.scalar('train/loss', loss_op)

        # Create a validation loss if possible.
        if validation_labels is not None:
            validation_labels = tf.reshape(validation_labels, [-1, 10])
            validation_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model_deploy["logits"], labels=validation_labels))
            tf.summary.scalar('dev/loss', validation_loss_op)

        return train_op
