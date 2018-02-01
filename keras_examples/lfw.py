from keras_models.lfw import deeplfw, prepare_data, create_triplets
from datasets.classification.named_folders import named_folders
from keras.optimizers import SGD
from keras.utils import plot_model
from utils.plot_losses import PlotLosses
import time
import datetime
import numpy as np
from datasets.tfrecords import PHASE_VALIDATION, PHASE_TRAIN
from datasets.batch_generator import batch_generator
import os

# Data can be downloaded here:
#   http://vis-www.cs.umass.edu/lfw/#download
# The lfw-deepfunneled version is recommended. However, the normal version might work too, if you design a better net.
#   http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
#   http://vis-www.cs.umass.edu/lfw/lfw.tgz

if __name__ == "__main__":
    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    model_output_folder = "keras_models/weights/deeplfw"
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    print("\nLoading Dataset...")
    roi = (200, 200)
    train_data = named_folders("data/lfw-deepfunneled", phase=PHASE_TRAIN, prepare_features=prepare_data, crop_roi=roi, no_split_folder=10)
    validation_data = named_folders("data/lfw-deepfunneled", phase=PHASE_VALIDATION, prepare_features=prepare_data, crop_roi=roi, no_split_folder=10)

    print("\nCreating Model: deeplfw")
    model = deeplfw()
    plot_model(model, to_file='keras_models/weights/deeplfw/architecture_%s.png' % time_str, show_shapes=True)
    print("Saved structure in: keras_models/weights/deeplfw/architecture_%s.png" % time_str)

    print("\nCreate SGD Optimizer")
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # TODO select correct loss here! (probably mse)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print("\nFit model...")
    plot_losses = PlotLosses("keras_models/weights/deeplfw/loss_%s.png" % time_str,
                             "keras_models/weights/deeplfw/acc_%s.png" % time_str,
                             validation_data=batch_generator(validation_data))

    model.fit_generator(batch_generator(create_triplets(train_data, model), batch_size=256), steps_per_epoch=20, nb_epoch=200, callbacks=[plot_losses], verbose=0)

    model_json = model.to_json()
    with open("keras_models/weights/deeplfw/%s.json" % time_str, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("keras_models/weights/deeplfw/%s.h5" % time_str)
