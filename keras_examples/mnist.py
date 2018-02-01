from keras_models.mnist_cnn import mnist_toy_net, prepare_data
from datasets.classification.mnist import mnist
from keras.optimizers import SGD
from keras.utils import plot_model
from utils.plot_losses import PlotLosses
import time
import datetime
import numpy as np


if __name__ == "__main__":
    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

    print("\nLoading Dataset...")
    images, labels = mnist("data/mnist", "train", prepare_data)
    test_images, test_labels = mnist("data/mnist", "test", prepare_data)

    print("Training Images: %d" % len(images))
    print("Test Images: %d" % len(test_images))

    print("\nCreating Model: mnist_toy_net")
    model = mnist_toy_net()
    plot_model(model, to_file='models/weights/mnist_toy_net_architecture_%s.png' % time_str, show_shapes=True)
    print("Saved structure in: models/weights/mnist_toy_net_architecture_%s.png" % time_str)

    print("\nCreate SGD Optimizer")
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print("\nFit model...")
    plot_losses = PlotLosses("models/weights/mnist_toy_net_loss_%s.png" % time_str,
                             "models/weights/mnist_toy_net_acc_%s.png" % time_str)
    model.fit(x=images, y=labels, batch_size=128, epochs=200, callbacks=[plot_losses], validation_data=(test_images, test_labels), verbose=0)

    model_json = model.to_json()
    with open("models/weights/mnist_toy_net_%s.json" % time_str, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/weights/mnist_toy_net_%s.h5" % time_str)

    classes = np.argmax(model.predict(test_images), axis=1)
    print(classes)
