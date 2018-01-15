from models.model import Model

class LFWNetwork(Model):
    def __init__(self, hyper_params_filepath):
        pass


    def setup(self, session):
        """
        Initialize everything for the model that needs a session.
        This includes loading checkpoints if provided in the hyperparameters.

        :param session: The tensorflow session to live inside.
        """
        pass

    def predict(self, features):
        """
        Predict the output of the network given only the feature input.
        This is handy for deployment of the network.

        :param features: The input features of the network. For a cnn this is an image.
        """
        pass

    def fit(self, training_data, iters, validation_data=None, summary_iters=1000, verbose=True):
        """
        Fit the model to given training data.

        :param training_data: training_data TODO
        :param validation_data: validation_data TODO (This data is optional, if not provided no validation is done.)
        :param iters: iters The number of epochs to train in total.
        :param summary_iters: summary_iters How many epochs to do between two summaries.
        :param verbose: verbose If you want debug outputs or not.
        """
        pass


    def export(self):
        """
        Export the model for deployment.

        The exported models can be used in an android app or a rosnode.
        """
        pass
