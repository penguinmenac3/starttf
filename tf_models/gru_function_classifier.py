import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


def create_model(hyper_params, input_tensor, reuse_weights=False, deploy_model=False, feed_dict={}):
    outputs = {}
    with tf.variable_scope('GruFunctionClassifier') as scope:
        if reuse_weights:
            scope.reuse_variables()

        # Define inputs
        input_tensor = tf.reshape(input_tensor, (hyper_params.train.batch_size, hyper_params.arch.sequence_length, 1))
        Hin = tf.placeholder(tf.float32,
                             [None, hyper_params.arch.hidden_layer_size * hyper_params.arch.hidden_layer_depth],
                             name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
        feed_dict[Hin] = np.zeros(
            [hyper_params.train.batch_size, hyper_params.arch.hidden_layer_size * hyper_params.arch.hidden_layer_depth])

        # Define the actual cells
        cells = [rnn.GRUCell(hyper_params.arch.hidden_layer_size) for _ in range(hyper_params.arch.hidden_layer_depth)]

        # "naive dropout" implementation
        if not deploy_model:
            cells = [rnn.DropoutWrapper(cell, input_keep_prob=hyper_params.arch.pkeep) for cell in cells]

        multicell = rnn.MultiRNNCell(cells, state_is_tuple=False)
        if not deploy_model:
            multicell = rnn.DropoutWrapper(multicell,
                                           output_keep_prob=hyper_params.arch.pkeep)  # dropout for the softmax layer

        Yr, H = tf.nn.dynamic_rnn(multicell, input_tensor, dtype=tf.float32, initial_state=Hin)
        H = tf.identity(H, name='H')  # just to give it a name

        # Softmax layer implementation:
        # Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, self.hyper_params.arch.output_dim ] => [ BATCHSIZE x SEQLEN, self.hyper_params.arch.output_dim ]
        # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
        # From the readout point of view, a value coming from a sequence time step or a minibatch item is the same thing.

        # Select last output.
        output = tf.transpose(Yr, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        outputs["logits"] = layers.linear(last, hyper_params.arch.output_dimension)
        outputs["probs"] = tf.nn.softmax(outputs["logits"], name="probs")
    return outputs, feed_dict


def create_loss(hyper_params, train_model, validation_model, train_labels, validation_labels=None):
    labels = tf.reshape(train_labels, [-1, hyper_params.arch.output_dimension])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_model["logits"], labels=labels))
    train_op = tf.train.RMSPropOptimizer(learning_rate=hyper_params.train.learning_rate,
                                         decay=hyper_params.train.decay).minimize(loss_op)
    tf.summary.scalar('train/loss', loss_op)

    # Create a validation loss if possible.
    if validation_labels is not None:
        validation_labels = tf.reshape(validation_labels, [-1, hyper_params.arch.output_dimension])
        validation_loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=validation_model["logits"], labels=validation_labels))
        tf.summary.scalar('validation/loss', validation_loss_op)

    return train_op
