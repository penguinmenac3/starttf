########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
#                                                                                      #
# Modified by: Michael Fuerst 2018                                                     #
########################################################################################

import tensorflow as tf
import numpy as np


def create_model(input_tensor, mode, hyper_params):
    outputs = {}

    with tf.variable_scope('vgg16') as scope:
        if mode == tf.estimator.ModeKeys.EVAL:
            scope.reuse_variables()

        parameters = []

        # zero-mean input
        with tf.variable_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = input_tensor - mean

        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 3, 64], name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[64],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 64, 64], name='weights')
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[64],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 64, 128], name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[128],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 128, 128], name='weights')
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[128],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 128, 256], name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[256],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 256, 256], name='weights')
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[256],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 256, 256], name='weights')
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[256],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 256, 512], name='weights')
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[512],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[512],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[512],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[512],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[512],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32,
                                     stddev=1e-1), shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0, dtype=tf.float32), shape=[512],
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name="relu")
            parameters += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # Put all relevant encoder layers into the api of the model.
        outputs["vgg16/conv1_1"] = conv1_1
        outputs["vgg16/conv1_2"] = conv1_2
        outputs["vgg16/pool1"] = pool1
        outputs["vgg16/conv2_1"] = conv2_1
        outputs["vgg16/conv2_2"] = conv2_2
        outputs["vgg16/pool2"] = pool2
        outputs["vgg16/conv3_1"] = conv3_1
        outputs["vgg16/conv3_2"] = conv3_2
        outputs["vgg16/conv3_3"] = conv3_3
        outputs["vgg16/pool3"] = pool3
        outputs["vgg16/conv4_1"] = conv4_1
        outputs["vgg16/conv4_2"] = conv4_2
        outputs["vgg16/conv4_3"] = conv4_3
        outputs["vgg16/pool4"] = pool4
        outputs["vgg16/conv5_1"] = conv5_1
        outputs["vgg16/conv5_2"] = conv5_2
        outputs["vgg16/conv5_3"] = conv5_3
        outputs["vgg16/pool5"] = pool5

        # If required add the head to the network.
        if not hyper_params.vgg16.encoder_only:
            # fc1
            with tf.variable_scope('fc1') as scope:
                shape = int(np.prod(pool5.get_shape()[1:]))
                fc1w = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32, stddev=1e-1),
                                       shape=[shape, 4096], name='weights')
                fc1b = tf.get_variable(initializer=tf.constant_initializer(1.0, dtype=tf.float32), shape=[4096],
                                       trainable=True, name='biases')
                pool5_flat = tf.reshape(pool5, [-1, shape])
                fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
                fc1 = tf.nn.relu(fc1l)
                parameters += [fc1w, fc1b]
                outputs["vgg16/fc1"] = fc1

            # fc2
            with tf.variable_scope('fc2') as scope:
                fc2w = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32, stddev=1e-1),
                                       shape=[4096, 4096], name='weights')
                fc2b = tf.get_variable(initializer=tf.constant_initializer(1.0, dtype=tf.float32), shape=[4096],
                                       trainable=True, name='biases')
                fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
                fc2 = tf.nn.relu(fc2l)
                parameters += [fc2w, fc2b]
                outputs["vgg16/fc2"] = fc2

            # fc3
            with tf.variable_scope('fc3') as scope:
                fc3w = tf.get_variable(initializer=tf.random_normal_initializer(mean=0, dtype=tf.float32, stddev=1e-1),
                                       shape=[4096, 1000], name='weights')
                fc3b = tf.get_variable(initializer=tf.constant_initializer(1.0, dtype=tf.float32), shape=[1000],
                                       trainable=True, name='biases')
                fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
                parameters += [fc3w, fc3b]
        
                probs = tf.nn.softmax(fc3l)
                outputs["logits"] = fc3l
                outputs["probs"] = probs

        outputs["vgg16/parameters"] = parameters
    return outputs


def load_weights(vgg_model, weight_file, session):
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        if i >= len(vgg_model["vgg16/parameters"]):
            print("Loaded all layers.")
            return
        print(i, k, np.shape(weights[k]))
        session.run(vgg_model["vgg16/parameters"][i].assign(weights[k]))
