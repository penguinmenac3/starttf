################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details: 
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
# Modified by Michael Fürst, 2018
################################################################################

import tensorflow as tf
import numpy as np


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


def create_model(input_tensor, mode, hyper_params):
    """
    An alexnet network which can load caffe converted weights.

    This model is just modified by the original authors are Michael Guerzhoy and Davi Frossard, 2016

    Either convert them yourself or download them from a third party.

    Convert weights using: https://github.com/ethereon/caffe-tensorflow
    Or download weights: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

    :param input_tensor: The input tensor dict containing a "image" rgb tensor.
    :param mode: Execution mode as a tf.estimator.ModeKeys
    :param hyper_params: The hyper param file.
    :return: A dictionary containing all output tensors.
    """
    weight_file = hyper_params.arch.weight_file
    #net_data = np.load(open(weight_file, "rb"), encoding="latin1").item()
    net_data = np.load(weight_file).item()
    model = {}

    with tf.variable_scope('alexnet') as scope:
        if mode == tf.estimator.ModeKeys.EVAL:
            scope.reuse_variables()
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = tf.Variable(net_data["conv1"][0], name="conv1W")
        conv1b = tf.Variable(net_data["conv1"][1], name="conv1b")
        conv1_in = conv(input_tensor["image"], conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in, name="conv1")

        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name="lrn1")

        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="pool1")

        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(net_data["conv2"][0], name="conv2W")
        conv2b = tf.Variable(net_data["conv2"][1], name="conv2b")
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in, name="conv2")

        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name="lrn2")

        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="pool2")

        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(net_data["conv3"][0], name="conv3W")
        conv3b = tf.Variable(net_data["conv3"][1], name="conv3b")
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in, name="conv3")

        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(net_data["conv4"][0], name="conv4W")
        conv4b = tf.Variable(net_data["conv4"][1], name="conv4b")
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in, name="conv4")

        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(net_data["conv5"][0], name="conv5W")
        conv5b = tf.Variable(net_data["conv5"][1], name="conv5b")
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in, name="conv5")

        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="pool5")

        #fc6
        #fc(4096, name='fc6')
        fc6W = tf.Variable(net_data["fc6"][0], name="fc6W")
        fc6b = tf.Variable(net_data["fc6"][1], name="fc6b")
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b, name="fc6")

        #fc7
        #fc(4096, name='fc7')
        fc7W = tf.Variable(net_data["fc7"][0], name="fc7W")
        fc7b = tf.Variable(net_data["fc7"][1], name="fc7b")
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name="fc7")

        #fc8
        #fc(1000, relu=False, name='fc8')
        fc8W = tf.Variable(net_data["fc8"][0], name="fc8W")
        fc8b = tf.Variable(net_data["fc8"][1], name="fc8b")
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b, name="fc8")

        #prob
        #softmax(name='prob'))
        prob = tf.nn.softmax(fc8, name="probs")

        # Put all relevant layers into the api of the model.
        model["conv1"] = conv1
        model["lrn1"] = lrn1
        model["pool1"] = maxpool1
        model["conv2"] = conv2
        model["lrn2"] = lrn2
        model["pool2"] = maxpool2
        model["conv3"] = conv3
        model["conv4"] = conv4
        model["conv5"] = conv5
        model["pool5"] = maxpool5
        model["fc6"] = fc6
        model["fc7"] = fc7
        model["logits"] = fc8
        model["probs"] = prob

    return model
