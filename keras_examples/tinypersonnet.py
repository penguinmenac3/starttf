from keras_models.tinypersonnet import tinypersonnet, prepare_data
from datasets.classification.named_folders import named_folders
from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model
import os
from utils.plot_losses import PlotLosses
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from datasets.batch_generator import batch_generator
from utils.dict2obj import json_file_to_object

DEBUG = False


if __name__ == "__main__":
    hyper_parameters_file = "keras_examples/tinypersonnet.json"
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
    roi = (100, 50)
    class_map = {"p": 0, "n": 1}
    train_data = named_folders("data/person_classification",
                               phase="train",
                               prepare_features=prepare_data,
                               class_idx=class_map,
                               crop_roi=roi)
    validation_data = named_folders("data/person_classification",
                                    phase="test",
                                    prepare_features=prepare_data,
                                    class_idx=class_map,
                                    crop_roi=roi)

    print("\nCreating Model: tinypersonnet")
    model = tinypersonnet()
    #plot_model(model, to_file=model_output_folder + '/architecture_%s.png' % time_str, show_shapes=True)
    print("Saved structure in: " + model_output_folder + "/architecture.png")

    print("\nCreate SGD Optimizer")
    if hyper_parameters.train.optimizer == "SGD":
        optimizer = SGD(lr=hyper_parameters.train.learning_rate, decay=hyper_parameters.train.decay, momentum=hyper_parameters.train.momentum, nesterov=True)
    elif hyper_parameters.train.optimizer == "RMSProp":
        optimizer = RMSprop(lr=hyper_parameters.train.learning_rate)
    else:
        raise RuntimeError("Unsupported optimizer: %s" % hyper_parameters.train.optimizer)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    print("\nFit model...")
    plot_losses = PlotLosses(model_output_folder + "/loss.png",
                             model_output_folder + "/acc.png",
                             model_output_folder + "/f1.png",
                             model_output_folder + "/roc.png",
                             validation_data=batch_generator(validation_data), f1_score_class=class_map["p"])
    model.fit_generator(batch_generator(train_data, batch_size=hyper_parameters.train.batch_size),
                        steps_per_epoch=hyper_parameters.train.summary_iters,
                        nb_epoch=int(hyper_parameters.train.iters / hyper_parameters.train.summary_iters),
                        callbacks=[plot_losses],
                        verbose=0)

    model_json = model.to_json()
    with open(model_output_folder + "/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_output_folder + "/model.h5")
