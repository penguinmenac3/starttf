import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D, Flatten, Lambda, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.initializers import Orthogonal
from starttf.modules import Module, Loss, Metrics
from starttf.train import HyperParams
from opendatalake.simple_sequence import SimpleSequence


class FashionMnistParams(HyperParams):
    def __init__(self):
        super().__init__()
        self.problem.number_of_categories = 10
        self.problem.base_dir = "datasets"

        self.train.epochs = 20
        self.train.l2_weight = 0.01
        self.train.batch_size = 64
        self.train.summary_steps = 50
        self.train.experiment_name = "%NAME%"
        self.train.checkpoint_path = "checkpoints"
        self.train.learning_rate.type = "exponential"
        self.train.learning_rate.start_value = 0.001
        self.train.learning_rate.end_value = 0.0001

        self.arch.model = "examples.fashion_mnist.FashionMnistModel"
        self.arch.loss = "examples.fashion_mnist.FashionMnistLoss"
        self.arch.metrics = "examples.fashion_mnist.FashionMnistMetrics"
        self.arch.prepare = "examples.fashion_mnist.FashionMnistDataset"


class FashionMnistDataset(SimpleSequence):
    def __init__(self, hyperparams, phase):
        super().__init__(hyperparams, phase)
        ((trainX, trainY), (valX, valY)) = fashion_mnist.load_data()
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.training = phase == "train"

    def num_samples(self):
        if self.training:
            return len(self.trainX)
        else:
            return len(self.valX)

    def get_sample(self, idx):
        label = np.zeros(shape=(10,), dtype="float32")
        if self.training:
            label[self.trainY[idx]] = 1
            return {"features": np.array(self.trainX[idx], dtype="float32")}, {"class_id": label}
        else:
            label[self.valY[idx]] = 1
            return {"features": np.array(self.valX[idx], dtype="float32")}, {"class_id": label}


class FashionMnistModel(Module):
    def __init__(self, name="FashionMnistModel"):
        super().__init__(name)
        l2_weight = self.hyperparams.train.l2_weight
        num_outputs = self.hyperparams.problem.number_of_categories
        self.layers = []
        self.layers.append(Lambda(lambda x: tf.keras.backend.expand_dims(x)))
        self.layers.append(BatchNormalization())
        self.layers.append(Conv2D(filters=12, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.layers.append(MaxPooling2D())

        self.layers.append(BatchNormalization())
        self.layers.append(Conv2D(filters=18, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.layers.append(MaxPooling2D())

        self.layers.append(BatchNormalization())
        self.layers.append(Conv2D(filters=18, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.layers.append(MaxPooling2D())

        self.layers.append(BatchNormalization())
        self.layers.append(Conv2D(filters=18, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.layers.append(GlobalAveragePooling2D())

        self.layers.append(BatchNormalization())
        self.layers.append(Dense(units=num_outputs, activation="softmax", kernel_initializer=Orthogonal()))

    def call(self, features, **ignored):
        net = features
        for l in self.layers:
            net = l(net)
        return {"class_id": net}


class FashionMnistLoss(Loss):
    def __init__(self):
        super().__init__(name="FashionMnistLoss")
        self.losses = {"class_id": categorical_crossentropy}


class FashionMnistMetrics(Metrics):
    def __init__(self):
        super().__init__(name="FashionMnistLoss")
        self.metrics = {"class_id": [categorical_crossentropy, categorical_accuracy, self.mse, self.variance_in_loss]}

    def categorical_variance_loss(self, y_true, y_pred):
        L = categorical_crossentropy(y_true, y_pred)

        mean = tf.reduce_mean(L)
        return tf.reduce_mean(L) + mean_squared_error(mean, L)

    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def variance_in_loss(self, y_true, y_pred):
        # Compute Loss
        L = categorical_crossentropy(y_true, y_pred)

        mean = tf.reduce_mean(L)
        return mean_squared_error(mean, L)
