import os
from scipy.misc import imread
from random import shuffle
import numpy as np
import json


def one_hot(idx, max_idx):
    label = np.zeros(max_idx)
    label[idx] = 1
    return label


def crop_center(img,cropy,cropx):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def named_folders(base_dir, phase, prepare_features=None, class_idx={}, crop_roi=None, file_extension=".png", overwrite_cashe=False):
    if phase is not None:
        classes_dir = os.path.join(base_dir, phase)
    if phase is None:
        classes_dir = base_dir
    classes = os.listdir(classes_dir)
    images = []
    labels = []

    if overwrite_cashe:
        if os.path.exists(os.path.join(classes_dir, "images.json")):
            os.remove(os.path.join(classes_dir, "images.json"))
        if os.path.exists(os.path.join(classes_dir, "labels.json")):
            os.remove(os.path.join(classes_dir, "labels.json"))

    if os.path.exists(os.path.join(classes_dir, "images.json")) and os.path.exists(os.path.join(classes_dir, "labels.json")):
        print("Using buffer files.")
        with open(os.path.join(classes_dir, "images.json"), 'r') as infile:
            images = json.load(infile)
        with open(os.path.join(classes_dir, "labels.json"), 'r') as infile:
            labels = json.load(infile)
    else:
        print("No buffer files found. Reading folder structure and creating buffer files.")
        for c in classes:
            if c == "labels.json" or c == "images.json": continue
            if c not in class_idx:
                class_idx[c] = len(class_idx)
            class_dir = os.path.join(classes_dir, c)
            for filename in os.listdir(class_dir):
                if filename.endswith(file_extension):
                    images.append(os.path.join(class_dir, filename))
                    labels.append(class_idx[c])

        print(images)
        with open(os.path.join(classes_dir, "images.json"), 'w') as outfile:
            json.dump(images, outfile)
        with open(os.path.join(classes_dir, "labels.json"), 'w') as outfile:
            json.dump(labels, outfile)

    image_idx = [x for x in range(len(images))]
    n_classes = len(classes)

    def gen():
        while True:
            # Shuffle data
            shuffle(image_idx)
            for idx in image_idx:
                feature = imread(images[idx], mode="RGB")
                if crop_roi is not None:
                    feature = crop_center(feature, crop_roi[0], crop_roi[1])
                if prepare_features:
                    feature = prepare_features(feature)
                yield (feature, one_hot(labels[idx], n_classes))

    return gen()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    roi = (200, 200)
    train_data = named_folders("data/lfw-deepfunneled", phase=None, crop_roi=roi, file_extension=".jpg")

    img, label = next(train_data)
    print("Image shape:")
    print(img.shape)

    for img, label in train_data:
        plt.imshow(img)
        plt.show()
