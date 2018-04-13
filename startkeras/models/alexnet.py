# Originally written (before refactor) by:
#   https://gist.github.com/JBed/c2fb3ce8ed299f197eff
# other implementation found here:
#   http://dandxy89.github.io/ImageModels/alexnet/
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from scipy.misc import imresize


def prepare_data(img):
    img = imresize(img, (224, 224)).astype(np.float32)
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]  # rgb2bgr
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def alexnet(weights_path=None):
    # caffe alexnet (no dual network architecture)
    model = Sequential()
    model.add(Convolution2D(64, 3, 11, 11, border_mode='full'))
    model.add(BatchNormalization((64, 226, 226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(128, 64, 7, 7, border_mode='full'))
    model.add(BatchNormalization((128, 115, 115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(192, 128, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128, 112, 112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(256, 192, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128, 108, 108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Flatten())
    model.add(Dense(12 * 12 * 256, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 1000, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
