import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

from tf_models.model import Model


class FunctionClassifier(Model):
    def __init__(self, hyper_params_filepath):
        super(FunctionClassifier, self).__init__(hyper_params_filepath)

    def _create_model(self, input_tensor, reuse_weights, validation=False):
        outputs = {}
        with tf.variable_scope('NeuralNet') as scope:
            if reuse_weights:
                scope.reuse_variables()

            input_tensor = tf.reshape(input_tensor, (self.hyper_params.train.batch_size, self.hyper_params.arch.sequence_length, 1))

            Hin = tf.placeholder(tf.float32, [None, self.hyper_params.arch.hidden_layer_size * self.hyper_params.arch.hidden_layer_depth], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
            self.feed_dict[Hin] = np.zeros([self.hyper_params.train.batch_size, self.hyper_params.arch.hidden_layer_size * self.hyper_params.arch.hidden_layer_depth])

            # using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
            # dynamic_rnn infers SEQLEN from the size of the inputs Xo

            # How to properly apply dropout in RNNs: see README.md
            cells = [rnn.GRUCell(self.hyper_params.arch.hidden_layer_size) for _ in range(self.hyper_params.arch.hidden_layer_depth)]
            # "naive dropout" implementation
            dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=self.hyper_params.arch.pkeep) for cell in cells]
            multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
            multicell = rnn.DropoutWrapper(multicell, output_keep_prob=self.hyper_params.arch.pkeep)  # dropout for the softmax layer

            Yr, H = tf.nn.dynamic_rnn(multicell, input_tensor, dtype=tf.float32, initial_state=Hin)
            # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
            # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

            H = tf.identity(H, name='H')  # just to give it a name

            # Softmax layer implementation:
            # Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, self.hyper_params.arch.output_dim ] => [ BATCHSIZE x SEQLEN, self.hyper_params.arch.output_dim ]
            # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
            # From the readout point of view, a value coming from a sequence time step or a minibatch item is the same thing.

            # Select last output.
            output = tf.transpose(Yr, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            #Yflat = tf.reshape(Yr, [-1, self.hyper_params.arch.hidden_layer_size])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
            outputs["logits"] = layers.linear(last, self.hyper_params.arch.output_dimension)     # [ BATCHSIZE x SEQLEN, self.hyper_params.arch.output_dim ]
        return outputs


    def _create_loss(self, labels, validation_labels=None):
        labels = tf.reshape(labels, [-1, self.hyper_params.arch.output_dimension])
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model_train["logits"], labels=labels))
        train_op = tf.train.RMSPropOptimizer(learning_rate=self.hyper_params.train.learning_rate, decay=self.hyper_params.train.decay).minimize(loss_op)

        # Create a validation loss if possible.
        validation_loss_op = None
        if validation_labels is not None:
            validation_labels = tf.reshape(validation_labels, [-1, self.hyper_params.arch.output_dimension])
            validation_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model_validation["logits_validation"], labels=validation_labels))

        return train_op, loss_op, validation_loss_op
