import json
import abc

from utils.dict2obj import Dict2Obj
import tensorflow as tf


class Model(object):
    """
    A general api to a neural network model.
    Every neural network model should be a subclass of this.
    """
    def __init__(self, hyper_params_filepath):
        """
        Create the model.

        :param hyper_params_filepath: The path to the hyper parameters file
        """
        self.hyper_params = Dict2Obj(json.load(open(hyper_params_filepath)))
        self.model_train = None
        self.model_deploy = None
        self.sess = None
        self.feed_dict = {}

    def setup(self, session):
        """
        Initialize everything for the model that needs a session.
        This includes loading checkpoints if provided in the hyperparameters.

        :param session: The tensorflow session to live inside.
        """
        self.sess = session

    @abc.abstractmethod
    def _create_model(self, input_tensor, reuse_weights, is_deploy_model=False):
        """
            Create a model.
        """
        return {}

    @abc.abstractmethod
    def _create_loss(self, labels, validation_labels=None):
        """
            Create a loss.
        """
        train_op, loss_op, validation_loss_op = None, None, None
        return train_op, loss_op, validation_loss_op

    def predict(self, features):
        """
        Predict the output of the network given only the feature input.
        This is handy for deployment of the network.

        :param features: The input features of the network. For a cnn this is an image.
        """
        pass

    def fit(self, features, labels, validation_examples_num=0, validation_features=None, validation_labels=None, verbose=True):
        """
        Fit the model to given training data.

        :param features: features An input queue tensor as provided by prepare_training.read_tf_records(...).
        :param labels: labels An input queue tensor as provided by prepare_training.read_tf_records(...).
        :param validation_features: validation_features An input queue tensor. (This data is optional, if not provided no validation is done.)
        :param validation_labels: validation_labels An input queue tensor. (This data is optional, if not provided no validation is done.)
        :param iters: iters The number of epochs to train in total.
        :param summary_iters: summary_iters How many epochs to do between two summaries.
        :param verbose: verbose If you want debug outputs or not.
        """
        self.model_train = self._create_model(features, reuse_weights=False)
        self.model_deploy = self._create_model(validation_features, reuse_weights=True, is_deploy_model=True)

        # Create loss and training op.
        train_op, loss_op, validation_loss_op = self._create_loss(labels, validation_labels)

        # Init vars.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        # Prepare training.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        saver = tf.train.Saver()

        # Train
        acc_loss = 0.0
        loss_normalizer = float(self.hyper_params.train.batch_size) / float(self.hyper_params.train.summary_iters)
        for i_step in range(self.hyper_params.train.iters):

            # Train step.
            _, loss = self.sess.run([train_op, loss_op], feed_dict=self.feed_dict)
            acc_loss += loss

            # Do validation and summary.
            if i_step % self.hyper_params.train.summary_iters == 0:
                loss_val = 0.0
                if validation_examples_num > 0:
                    for i_val in range(int(validation_examples_num / self.hyper_params.train.batch_size)):
                        loss_val_loc = self.sess.run([validation_loss_op], feed_dict=self.feed_dict)[0]
                        loss_val += loss_val_loc

                if verbose:
                    print("Iter: %d, Loss: %.4f, Validation Loss: %.4f" % (i_step, acc_loss / loss_normalizer, loss_val / loss_normalizer))
                    acc_loss = 0.0

                saver.save(self.sess, self.hyper_params.train.checkpoint_path, global_step=i_step)

        if verbose:
            print("Training stopped.")

        coord.request_stop()
        coord.join(threads)

    def export(self):
        """
        Export the model for deployment.

        The exported models can be used in an android app or a rosnode.
        """
        pass
