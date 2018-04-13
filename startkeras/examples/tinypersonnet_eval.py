from keras_models.tinypersonnet import tinypersonnet, prepare_data
from datasets.classification.named_folders import crop_center
from keras.optimizers import SGD
import time
import datetime
import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt


if __name__ == "__main__":
    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

    print("\nLoading Dataset...")
    roi = (100, 50)
    class_map = {"p": 0, "n": 1}
    folder = "data/person_classification_test/test"
    test_image_files = "data/person_classification_test/testFiles.txt"
    test_result_file = "data/person_classification_test/result.txt"
    images = []
    with open(test_image_files, "r") as file:
        img_names = file.read().split("\n")[:-1]  # remove last line since file usually ends on a newline...
        for img in img_names:
            if img == "":
                print("Empty line (is there an error in your input data?")
                continue
            img_path = os.path.join(folder, img.strip())
            feature = imread(img_path, mode="RGB")
            feature = crop_center(feature, roi[0], roi[1])
            feature = prepare_data(feature)
            images.append(feature)


    print("\nLoad Model: tinypersonnet")
    model = tinypersonnet(weights_path="models/weights/tinypersonnet_2018-01-09_20.04.47.h5")

    print("\nCreate SGD Optimizer")
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print("\nPredict images")
    predictions = model.predict(np.array(images))
    results = []
    positives = 0
    negatives = 0
    with open(test_result_file, "w") as text_file:
        for i in range(predictions.shape[0]):
            result = predictions[i][0]
            results.append(result)
            text_file.write("%.2f\n" % result)
            if result > 0.5:
                positives += 1
            else:
                negatives += 1
            plt.title("Result: %.2f" %result)
            img = images[i]
            img[:, :, 0] += 103.939
            img[:, :, 1] += 116.779
            img[:, :, 2] += 123.68
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]  # rgb2bgr
            img /= 255.0
            #print(img)
            #plt.imshow(img)
            #plt.show()

    print(results)
    print("positives: %d; negatives: %d" % (positives, negatives))