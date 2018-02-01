# A fully convolutional person detector for small images and few training data.
import numpy as np
from scipy.misc import imresize
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.regularizers import l2


def prepare_data(img):
    img = imresize(img, (200, 200)).astype(np.float32)
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]  # rgb2bgr
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img


def create_triplets(train_images, train_labels, model):
    triplet_in, triplet_out = train_images, train_labels

    print("TODO implement this")  # TODO implement this

    return triplet_in, triplet_out


def deeplfw(weights_path=None):
    l2_weight = 0.0
    x = Input(shape=(200, 200, 3))

    conv1 = x
    conv1 = Conv2D(16, (3, 3), activation='relu', name="conv1", kernel_regularizer=l2(l2_weight))(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(conv1)

    conv2 = pool1
    conv2 = Conv2D(32, (3, 3), activation='relu', name="conv2", kernel_regularizer=l2(l2_weight))(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(conv2)

    conv3 = pool2
    conv3 = Conv2D(64, (3, 3), activation='relu', name="conv3", kernel_regularizer=l2(l2_weight))(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = pool3
    conv4 = Conv2D(64, (3, 3), activation='relu', name="conv4", kernel_regularizer=l2(l2_weight))(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(conv4)
    pool4 = Dropout(0.5)(pool4)

    fc1 = Conv2D(64, (1, 1), activation='relu', name="fc1", kernel_regularizer=l2(l2_weight))(pool4)
    fc1 = Dropout(0.5)(fc1)

    face_vec = Flatten()(fc1)
    face_vec = Dense(1024, name="face_vec")(face_vec)

    # TODO define useful loss... and triplet usage

    # Create a keras model
    model = Model(inputs=[x], outputs=[face_vec])
    if weights_path:
        model.load_weights(weights_path)

    return model
