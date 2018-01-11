import os
from random import shuffle
import pickle
import numpy as np


# Here the cifar data can be downloaded.
CIFAR_10_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
# Extract it into the data/cifar-x folder
# For cifar-10 the data/cifar-10 folder should contain a data_batch_1, ... and a test_batch file.
# For cifar-100 the data/cifar-100 folder should contain a train and a test file.


def cifar(base_dir, phase, version=10):
    images = []
    labels = []

    if version == 10:
        if phase == "train":
            for x in range(1,5,1):
                data_path = os.path.join(base_dir, "data_batch_" + str(x))
                with open(data_path, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    images.extend(dict[b"data"])
                    labels.extend(dict[b"labels"])
        if phase == "test":
            data_path = os.path.join(base_dir, "test_batch")
            with open(data_path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                images.extend(dict[b"data"])
                labels.extend(dict[b"labels"])
    if version == 100:
        data_path = os.path.join(base_dir, phase)
        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            images.extend(dict[b"data"])
            labels.extend(dict[b"fine_labels"])

    image_idx = [x for x in range(len(images))]

    images_per_class = {}
    for label in labels:
        if str(label) not in images_per_class:
            images_per_class[str(label)] = 0
        images_per_class[str(label)] += 1

    def gen():
        while True:
            # Shuffle data
            shuffle(image_idx)
            for idx in image_idx:
                img = np.reshape(images[idx], (3, 32, 32))
                yield (img.transpose((1, 2, 0)), labels[idx])

    return images_per_class, gen()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    imgs_per_class, gen = cifar("data/cifar-10", "train")

    print("Classes and image count:")
    print(imgs_per_class)

    img, class_name = next(gen)
    print("Image shape:")
    print(img.shape)

    for img, class_name in gen:
        plt.imshow(img)
        plt.show()
