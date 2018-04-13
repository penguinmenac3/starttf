# A fully convolutional mnist net.
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.regularizers import l2


def prepare_data(img):
    img = np.reshape(np.array(img), (-1, 28, 28, 1))
    img[:, :] -= 0.5
    return img


def mnist_toy_net(weights_path=None):
    l2_weight = 0.0
    x = Input(shape=(28, 28, 1))

    conv1 = x
    conv1 = Conv2D(16, (3, 3), activation='relu', name="conv1", kernel_regularizer=l2(l2_weight))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu', name="conv2", kernel_regularizer=l2(l2_weight))(conv1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(conv2)

    conv3 = pool2
    conv3 = Conv2D(32, (3, 3), activation='relu', name="conv3", kernel_regularizer=l2(l2_weight))(conv3)
    conv4 = Conv2D(32, (3, 3), activation='relu', name="conv4", kernel_regularizer=l2(l2_weight))(conv3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(conv4)
    pool4 = Dropout(0.5)(pool4)

    probs = Flatten()(pool4)
    probs = Dense(10, activation="softmax", name="probs")(probs)

    # Create a keras model
    model = Model(inputs=[x], outputs=[probs])
    if weights_path:
        model.load_weights(weights_path)

    return model
