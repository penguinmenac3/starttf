from keras_models.lfw import deeplfw, prepare_data, create_triplets
from datasets.classification.named_folders import named_folders
from keras.optimizers import SGD
from keras.utils import plot_model
from utils.plot_losses import PlotLosses
import time
import datetime
import numpy as np

# Data can be downloaded here:
#   http://vis-www.cs.umass.edu/lfw/#download
# The lfw-deepfunneled version is recommended. However, the normal version might work too, if you design a better net.
#   http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
#   http://vis-www.cs.umass.edu/lfw/lfw.tgz

if __name__ == "__main__":
    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

    print("\nLoading Dataset...")
    roi = (200, 200)
    imgs_per_class, images, labels, class_idx = named_folders("data/lfw-deepfunneled", phase=None, prepare_features=prepare_data, crop_roi=roi)
    # TODO  Do a usefull test split instead of loading the same data twice!!!
    test_imgs_per_class, test_images, test_labels, test_class_idx = named_folders("data/lfw-deepfunneled", phase=None, prepare_features=prepare_data, crop_roi=roi)

    print("Train Classes and image count:")
    print(imgs_per_class)

    print("Test Classes and image count:")
    print(test_imgs_per_class)

    print("\nCreating Model: deeplfw")
    model = deeplfw()
    plot_model(model, to_file='models/weights/deeplfw_architecture_%s.png' % time_str, show_shapes=True)
    print("Saved structure in: models/weights/deeplfw_architecture_%s.png" % time_str)

    print("\nCreate SGD Optimizer")
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # TODO select correct loss here! (probably mse)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print("\nFit model...")
    plot_losses = PlotLosses("models/weights/deeplfw_loss_%s.png" % time_str,
                             "models/weights/deeplfw_acc_%s.png" % time_str,
                             validation_data=(test_images, test_labels))

    triplet_in, triplet_out = create_triplets(images, labels, model)
    model.fit(x=triplet_in, y=triplet_out, batch_size=64, epochs=200, callbacks=[plot_losses], validation_data=(test_images, test_labels), verbose=0)

    model_json = model.to_json()
    with open("models/weights/deeplfw_%s.json" % time_str, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/weights/deeplfw_%s.h5" % time_str)

    classes = np.argmax(model.predict(test_images), axis=1)
    print(classes)
