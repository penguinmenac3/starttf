import tensorflow as tf


def create_model(hyper_params, input_tensor, reuse_weights=False, deploy_model=False, feed_dict={}):
    outputs = {}
    with tf.variable_scope('NeuralNet') as scope:
        if reuse_weights:
            scope.reuse_variables()

        # TODO define net architecture.

        outputs["logits"] = input_tensor
    return outputs, feed_dict


def create_loss(hyper_params, train_model, validation_model, train_labels, validation_labels=None):
    reports = []
    labels = tf.reshape(train_labels, [-1, hyper_params.arch.output_dimension])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_model["logits"], labels=labels))
    train_op = tf.train.RMSPropOptimizer(learning_rate=hyper_params.train.learning_rate,
                                         decay=hyper_params.train.decay).minimize(loss_op)
    tf.summary.scalar('train/loss', loss_op)
    reports.append(loss_op)

    # Create a validation loss if possible.
    if validation_labels is not None:
        validation_labels = tf.reshape(validation_labels, [-1, hyper_params.arch.output_dimension])
        validation_loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=validation_model["logits"], labels=validation_labels))
        tf.summary.scalar('validation/loss', validation_loss_op)
        reports.append(validation_loss_op)

    return train_op, reports
