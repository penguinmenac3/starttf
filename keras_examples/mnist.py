from keras_models.mnist_cnn import mnist_toy_net, prepare_data
from datasets.classification.mnist import mnist
from keras.optimizers import SGD
from keras.utils import plot_model
from utils.plot_losses import KerasPlotLosses
from utils.dict2obj import json_file_to_object
import time
import datetime
import numpy as np
import os
from datasets.batch_generator import batch_generator


if __name__ == "__main__":
    hyper_parameters_file = "keras_examples/mnist.json"
    # Read hyperparameters
    hyper_parameters = json_file_to_object(hyper_parameters_file)

    # Create checkpoint folder and store hyper parameters there.
    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    model_output_folder = hyper_parameters.train.checkpoint_path + "/" + time_str
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)
    with open(model_output_folder + "/hyperparameters.json", "w") as json_file:
        with open(hyper_parameters_file, "r") as f:
            json_file.write(f.read())

    print("\nLoading Dataset...")
    train_data = mnist("data/mnist", "train", prepare_data)
    validation_data = mnist("data/mnist", "test", prepare_data)

    print("\nCreating Model: mnist_toy_net")
    model = mnist_toy_net()
    #plot_model(model, to_file=model_output_folder + '/architecture_%s.png' % time_str, show_shapes=True)
    print("Saved structure in: " + model_output_folder + "/architecture.png")

    print("\nCreate SGD Optimizer")
    sgd = SGD(lr=hyper_parameters.train.learning_rate, decay=hyper_parameters.train.decay, momentum=hyper_parameters.train.momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print("\nFit model...")
    plot_losses = KerasPlotLosses(model_output_folder + "/loss.png",
                                  model_output_folder + "/acc.png",
                                  validation_data=batch_generator(validation_data))
    model.fit_generator(batch_generator(train_data, batch_size=hyper_parameters.train.batch_size),
                        steps_per_epoch=hyper_parameters.train.summary_iters,
                        nb_epoch=int(hyper_parameters.train.iters / hyper_parameters.train.summary_iters),
                        callbacks=[plot_losses],
                        verbose=0)

    model_json = model.to_json()
    with open(model_output_folder + "/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_output_folder + "/model.h5")
