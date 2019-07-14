import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D, Flatten, Lambda, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.initializers import Orthogonal
import ailab
from ailab import PHASE_TRAIN, PHASE_VALIDATION
from ailab.experiment import Config
from ailab.data import DataProvider, DataProviderPipeline, BatchedDataProvider

import starttf as stf

class FashionMnistConfig(Config):
    def __init__(self):
        super().__init__()
        self.problem.number_of_categories = 10
        self.problem.base_dir = "datasets"
        #self.problem.tf_records_path = "tfrecords"

        self.train.epochs = 20
        self.train.l2_weight = 0.01
        self.train.batch_size = 32
        self.train.log_steps = 100
        self.train.experiment_name = "FashionMNIST"
        self.train.checkpoint_path = "checkpoints"
        self.train.learning_rate.type = "exponential"
        self.train.learning_rate.start_value = 0.001
        self.train.learning_rate.end_value = 0.0001

        self.arch.model = FashionMnistModel
        self.arch.loss = FashionMnistLoss
        self.arch.metrics = FashionMnistMetrics
        self.arch.prepare = DataProviderPipeline(FashionMnistDataset, BatchedDataProvider)


class FashionMnistDataset(DataProvider):
    def __init__(self, config, phase):
        super().__init__(config, phase)
        ((trainX, trainY), (valX, valY)) = fashion_mnist.load_data()
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.training = phase == PHASE_TRAIN

    def __len__(self):
        if self.training:
            return len(self.trainX)
        else:
            return len(self.valX)

    def __getitem__(self, idx):
        label = np.zeros(shape=(10,), dtype="float32")
        if self.training:
            label[self.trainY[idx]] = 1
            return {"features": np.array(self.trainX[idx], dtype="float32")}, {"class_id": label}
        else:
            label[self.valY[idx]] = 1
            return {"features": np.array(self.valX[idx], dtype="float32")}, {"class_id": label}


class FashionMnistModel(stf.Model):
    def __init__(self, name="FashionMnistModel"):
        super().__init__(name)
        l2_weight = ailab.config.train.l2_weight
        num_outputs = ailab.config.problem.number_of_categories
        self.linear = []
        self.linear.append(Lambda(lambda x: tf.keras.backend.expand_dims(x)))
        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=12, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_regularizer=l2(l2_weight), kernel_size=(3, 3),
                                  padding="same", activation="relu", kernel_initializer=Orthogonal()))
        self.linear.append(GlobalAveragePooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Dense(units=num_outputs, activation="softmax", kernel_initializer=Orthogonal()))

    @tf.function
    def forward(self, features, training=False):
        net = features
        for l in self.linear:
            net = l(net)
        return {"class_id": net}


class FashionMnistLoss(stf.Loss):
    def __init__(self):
        super().__init__(name="FashionMnistLoss")
        self.losses = {"class_id": categorical_crossentropy}


class FashionMnistMetrics(stf.Metrics):
    def __init__(self):
        super().__init__(name="FashionMnistLoss")
        self.metrics = {"class_id": [categorical_crossentropy, categorical_accuracy, self.mse, self.variance_in_loss]}

    @tf.function
    def categorical_variance_loss(self, y_true, y_pred):
        L = categorical_crossentropy(y_true, y_pred)

        mean = tf.reduce_mean(L)
        return tf.reduce_mean(L) + mean_squared_error(mean, L)

    @tf.function
    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @tf.function
    def variance_in_loss(self, y_true, y_pred):
        # Compute Loss
        L = categorical_crossentropy(y_true, y_pred)

        mean = tf.reduce_mean(L)
        return mean_squared_error(mean, L)


if __name__ == "__main__":
    #stf.modules.log_inputs = True
    #stf.modules.log_calls = True
    #stf.modules.log_creations = True

    config = FashionMnistConfig()
    stf.fit_supervised(config)
