# Originally written (before refactor) by:
#   https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14
# Weights:
#   https://drive.google.com/open?id=0B319laiAPjU3RE1maU9MMlh2dnc
from scipy.misc import imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from models.googlenet_custom_layers import PoolHelper, LRN
import numpy as np


def prepare_data(img):
    img = imresize(img, (224, 224)).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]  # rgb2bgr
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def inception_module(x, name, k1, k3, k5, kpool, W_norm):
    i1 = Convolution2D(k1[0], 1, 1, border_mode='same', activation='relu', name=name + '/1x1', W_regularizer=W_norm)(x)

    i3 = Convolution2D(k3[0], 1, 1, border_mode='same', activation='relu', name=name + '/3x3_reduce', W_regularizer=W_norm)(x)
    i3 = Convolution2D(k3[1], 3, 3, border_mode='same', activation='relu', name=name + '/3x3', W_regularizer=W_norm)(i3)

    i5 = Convolution2D(k5[0], 1, 1, border_mode='same', activation='relu', name=name + '/5x5_reduce', W_regularizer=W_norm)(x)
    i5 = Convolution2D(k5[1], 5, 5, border_mode='same', activation='relu', name=name + '/5x5', W_regularizer=W_norm)(i5)

    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name=name + '/pool')(x)
    pool = Convolution2D(kpool, 1, 1, border_mode='same', activation='relu', name=name + '/pool_proj', W_regularizer=W_norm)(pool)

    return merge([i1, i3, i5, pool], mode='concat', concat_axis=1, name=name + '/output')


def pooling_module(x, name):
    pool = ZeroPadding2D(padding=(1, 1))(x)
    pool = PoolHelper()(pool)
    return MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name=name + '/3x3_s2')(pool)


def auxiliary_classifier_module(x, outputs, name, W_norm):
    x = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name=name + '/ave_pool')(x)
    x = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name=name + '/conv', W_regularizer=W_norm)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name=name + '/fc', W_regularizer=W_norm)(x)
    x = Dropout(0.7)(x)
    x = Dense(outputs, name=name + '/classifier', W_regularizer=W_norm)(x)
    return Activation('softmax')(x)


def googlenet(weights_path=None, get_output_layers=False):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    input = Input(shape=(3, 224, 224))
    output_classes = 1000
    W_norm = l2(0.0002)

    # Conv layer 1 (+ pool, lrn)
    conv1 = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', activation='relu', name='conv1/7x7_s2', W_regularizer=W_norm)(input)
    pool1 = pooling_module(conv1, "pool1")
    pool1_norm1 = LRN(name='pool1/norm1')(pool1)

    # Conv layer 2 (+ lrn, pool)
    conv2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='conv2/3x3_reduce', W_regularizer=W_norm)(pool1_norm1)
    conv2 = Convolution2D(192, 3, 3, border_mode='same', activation='relu', name='conv2/3x3', W_regularizer=W_norm)(conv2)
    conv2_norm2 = LRN(name='conv2/norm2')(conv2)
    pool2 = pooling_module(conv2_norm2, "pool2")

    # Inception modules 3
    inception_3a = inception_module(pool2, "inception_3a", k1=(64,), k3=(96, 128), k5=(16, 32), kpool=32, W_norm=W_norm)
    inception_3b = inception_module(inception_3a, "inception_3b", k1=(128,), k3=(128, 192), k5=(32, 96), kpool=64, W_norm=W_norm)
    pool3 = pooling_module(inception_3b, "pool3")

    # Inception modules 4
    inception_4a = inception_module(pool3, "inception_4a", k1=(192,), k3=(96, 208), k5=(16, 48), kpool=64, W_norm=W_norm)
    inception_4b = inception_module(inception_4a, "inception_4b", k1=(160,), k3=(112, 224), k5=(24, 64), kpool=64, W_norm=W_norm)
    inception_4c = inception_module(inception_4b, "inception_4b", k1=(128,), k3=(128, 256), k5=(24, 64), kpool=64, W_norm=W_norm)
    inception_4d = inception_module(inception_4c, "inception_4d", k1=(112,), k3=(144, 288), k5=(32, 64), kpool=64, W_norm=W_norm)
    inception_4e = inception_module(inception_4d, "inception_4e", k1=(256,), k3=(160, 320), k5=(32, 128), kpool=128, W_norm=W_norm)
    pool4 = pooling_module(inception_4e, "pool4")

    # Inception modules 5
    inception_5a = inception_module(pool4, "inception_5a", k1=(256,), k3=(160, 320), k5=(32, 128), kpool=128, W_norm=W_norm)
    inception_5b = inception_module(inception_5a, "inception_5b", k1=(384,), k3=(192, 384), k5=(48, 128), kpool=128, W_norm=W_norm)

    # Auxiliary classifiers for training (inception_4a, inception_4d)
    auxiliary_classifier_1 = auxiliary_classifier_module(inception_4a, output_classes, "loss1", W_norm=W_norm)
    auxiliary_classifier_2 = auxiliary_classifier_module(inception_4d, output_classes, "loss2", W_norm=W_norm)

    # Classifier
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b)
    loss3_flat = Flatten()(pool5_7x7_s1)
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
    class_probs = Dense(output_classes, name='loss3/classifier', W_regularizer=W_norm)(pool5_drop_7x7_s1)
    class_probs = Activation('softmax', name='prob')(class_probs)

    # Create a keras model
    model = Model(input=input, output=[auxiliary_classifier_1, auxiliary_classifier_2, class_probs])

    if weights_path:
        model.load_weights(weights_path)

    if get_output_layers:
        layers = {"input": input, "mixed4a": inception_4a, "mixed4d": inception_4d, "mixed5b": inception_5b, "probs": class_probs}
        return model, layers

    return model
