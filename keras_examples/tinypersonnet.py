from keras_models.tinypersonnet import tinypersonnet, prepare_data
from datasets.classification.named_folders import named_folders
from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model
from utils.plot_losses import PlotLosses
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False


if __name__ == "__main__":
    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

    print("\nLoading Dataset...")
    roi = (100, 50)
    class_map = {"p": 0, "n": 1}
    imgs_per_class, images, labels, class_idx = named_folders("data/person_classification", "train", prepare_data, class_map, crop_roi=roi)
    test_imgs_per_class, test_images, test_labels, test_class_idx = named_folders("data/person_classification", "test", prepare_data, class_map, crop_roi=roi)

    if DEBUG:
        for idx in range(10):
            image = images[idx + 2436 - 5]
            print(image.shape)
            image[:, :, 0] += 103.939
            image[:, :, 1] += 116.779
            image[:, :, 2] += 123.68
            image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]  # rgb2bgr
            image /= 255.0
            plt.imshow(image)
            plt.show()

    print("Train Classes and image count:")
    print(imgs_per_class)

    print("Test Classes and image count:")
    print(test_imgs_per_class)

    print("\nCreating Model: tinypersonnet")
    model = tinypersonnet()
    plot_model(model, to_file='models/weights/tinypersonnet_architecture_%s.png' % time_str, show_shapes=True)
    print("Saved structure in: models/weights/tinypersonnet_architecture_%s.png" % time_str)

    print("\nCreate SGD Optimizer")
    #optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    print("\nFit model...")
    plot_losses = PlotLosses("models/weights/tinypersonnet_loss_%s.png" % time_str,
                             "models/weights/tinypersonnet_acc_%s.png" % time_str,
                             "models/weights/tinypersonnet_f1_%s.png" % time_str,
                             "models/weights/tinypersonnet_roc_%s.png" % time_str,
                             validation_data=(test_images, test_labels), f1_score_class=class_map["p"])
    model.fit(x=images, y=labels, batch_size=256, epochs=200, callbacks=[plot_losses], validation_data=(test_images, test_labels), verbose=0)

    model_json = model.to_json()
    with open("models/weights/tinypersonnet_%s.json" % time_str, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/weights/tinypersonnet_%s.h5" % time_str)

    classes = np.argmax(model.predict(test_images), axis=1)
    print(classes)
