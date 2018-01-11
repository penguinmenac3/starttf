import os
from scipy.misc import imread
from random import shuffle
import numpy as np


def one_hot(idx, max_idx):
    label = np.zeros(max_idx)
    label[idx] = 1
    return label


def crop_center(img,cropy,cropx):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def named_folders(base_dir, phase, prepare_features=None, class_idx={}, crop_roi=None, file_extension=".png"):
    if phase is not None:
        classes_dir = os.path.join(base_dir, phase)
    if phase is None:
        classes_dir = base_dir
    classes = os.listdir(classes_dir)
    images = []
    labels = []
    imgs_per_class = {}
    for c in classes:
        if c not in class_idx:
            class_idx[c] = len(class_idx)
        imgs_per_class[c] = 0
        class_dir = os.path.join(classes_dir, c)
        for filename in os.listdir(class_dir):
            if filename.endswith(file_extension):
                feature = imread(os.path.join(class_dir, filename), mode="RGB")
                if crop_roi is not None:
                    feature = crop_center(feature, crop_roi[0], crop_roi[1])
                if prepare_features:
                    feature = prepare_features(feature)
                images.append(feature)
                labels.append(one_hot(class_idx[c], len(classes)))
                imgs_per_class[c] += 1

    return imgs_per_class, np.array(images), np.array(labels), class_idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    roi = (200, 200)
    imgs_per_class, images, labels, class_idx = named_folders("data/lfw-deepfunneled", phase=None, crop_roi=roi, file_extension=".jpg")

    print("Classes and image count:")
    print(imgs_per_class)

    print("Image shape:")
    print(images[0].shape)

    print(labels[-1])

    i = 0
    for img, class_name in zip(images, labels):
        i += 1
        if i < 2430:
            continue
        print(class_name)
        plt.imshow(img)
        plt.show()
