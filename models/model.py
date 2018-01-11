import json
import abc

from utils.dict2obj import Dict2Obj


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

    @abc.abstractmethod
    def setup(self, session):
        """
        Initialize everything for the model that needs a session.
        This includes loading checkpoints if provided in the hyperparameters.

        :param session: The tensorflow session to live inside.
        """
        pass

    @abc.abstractmethod
    def predict(self, features):
        """
        Predict the output of the network given only the feature input.
        This is handy for deployment of the network.

        :param features: The input features of the network. For a cnn this is an image.
        """
        pass

    @abc.abstractmethod
    def fit(self, training_data, epochs, validation_data=None, summary_iters=1000, verbose=True):
        """
        Fit the model to given training data.

        :param training_data: TODO
        :param validation_data: TODO (This data is optional, if not provided no validation is done.)
        :param epochs: The number of epochs to train in total.
        :param summary_iters: How many epochs to do between two summaries.
        :param verbose: If you want debug outputs or not.
        """
        pass


    @abc.abstractmethod
    def export(self):
        """
        Export the model for deployment.

        The exported models can be used in an android app or a rosnode.
        """
        pass
