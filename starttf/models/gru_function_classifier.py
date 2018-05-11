import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


def create_model(input_tensor, mode, hyper_params):
    """
    Creates a function classifier model using a gru.

    :param input_tensor: A dictionary containing all input tensors.
    :param mode: If the network is training or evaluating (tf.estimator.ModeKeys)
    :param hyper_params: The hyper parameters object containing {"arch": {"pkeep": Float, "sequence_length": Int,
        "hidden_layer_depth": Int, "hidden_layer_size": Int, "output_dimension": Int}}
    :return: The model as a dictionary of output tensors.
    """
    outputs = {}
    with tf.variable_scope('GruFunctionClassifier') as scope:
        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
            scope.reuse_variables()

        batch_size = hyper_params.train.batch_size
        if mode == tf.estimator.ModeKeys.EVAL:
            batch_size = hyper_params.train.validation_batch_size
        if mode == tf.estimator.ModeKeys.PREDICT:
            batch_size = 1

        # Define inputs
        input_tensor = tf.reshape(input_tensor["feature"], (batch_size, hyper_params.arch.sequence_length, 1))
        Hin = tf.zeros([batch_size, hyper_params.arch.hidden_layer_size * hyper_params.arch.hidden_layer_depth], tf.float32, name="Hin")

        # Define the actual cells
        cells = [rnn.GRUCell(hyper_params.arch.hidden_layer_size) for _ in range(hyper_params.arch.hidden_layer_depth)]

        # "naive dropout" implementation
        if mode == tf.estimator.ModeKeys.TRAIN:
            cells = [rnn.DropoutWrapper(cell, input_keep_prob=hyper_params.arch.pkeep) for cell in cells]

        multicell = rnn.MultiRNNCell(cells, state_is_tuple=False)
        if mode == tf.estimator.ModeKeys.TRAIN:
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
    return outputs
