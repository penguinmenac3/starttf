from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def mnist(base_dir, phase, prepare_features=None):
    _mnist = input_data.read_data_sets(base_dir, one_hot=True)
    if phase == "test":
        images = _mnist.test.images
        labels = _mnist.test.labels
    else:
        images = _mnist.train.images
        labels = _mnist.train.labels
        
    if prepare_features is not None:
        images = prepare_features(images)
    return images, np.array(labels)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    images, labels = mnist("data/mnist", "train", lambda x: np.reshape(np.array(x), (-1, 28, 28)))

    print("Image shape:")
    print(images[0].shape)

    for img, class_name in zip(images, labels):
        plt.imshow(img)
        plt.show()
