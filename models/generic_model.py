from models.model import Model


class GenericModel(Model):
    def __init__(self, hyper_params_filepath, create_model_fn, create_loss_fn):
        """
        Create a generic model by passing in a create model function and a create loss function.
        :param hyper_params_filepath: The hyperparameters filepath like for every other model.
            This is then loaded into a hyperparams object.
        :param create_model_fn: A function that gets in the input_tensor and if it should reuse weights
            (and hyperparams object) and outputs the output tensors.
            (outputs = create_model_fn(input_tensor, reuse_weights, hyperparams)
        :param create_loss_fn: A function that gets in the output tensors of the create model the labels and the
            validation labels (and hyperparams object). It must return the training operation, the loss operation and
            the validation loss operation.
            (train_op, loss_op, validation_loss_op = create_loss_fn(outputs, labels, validation_labels, hyperparams))
        """
        super(GenericModel, self).__init__(hyper_params_filepath)
        self.__create_model = create_model_fn
        self.__create_loss = create_loss_fn

    def _create_model(self, input_tensor, reuse_weights):
        self.outputs = self.__create_model(input_tensor, reuse_weights, self.hyper_params)

    def _create_loss(self, labels, validation_labels=None):
        train_op, loss_op, validation_loss_op = self.__create_loss(self.outputs, labels, validation_labels, self.hyper_params)
        return train_op, loss_op, validation_loss_op
