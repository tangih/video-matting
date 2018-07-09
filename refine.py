import tensorflow as tf
import numpy as np


def init_conv(win_h, win_w, n_inputs, n_filters):
    """ creates initial tensor using Xavier method """
    n = win_h * win_w * int(n_inputs)
    std = np.sqrt(2. / n)
    conv = np.random.normal(loc=0., scale=std, size=(win_h, win_w, n_inputs, n_filters)).astype(np.float32)
    biases = np.random.normal(loc=0., scale=std, size=n_filters).astype(np.float32)
    return tf.get_variable(initializer=conv, name='weights'), tf.get_variable(initializer=biases, name='WOOO')


class RefineNet:
    def __init__(self):
        pass

    def conv_layer(self, input, n_filters, name):
        """ creates new convolution using Xavier initialisation """
        with tf.variable_scope(name):
            conv_weights, conv_biases = init_conv(3, 3, input.shape[-1], n_filters)
            conv = tf.nn.conv2d(input, conv_weights, [1, 1, 1, 1], padding='SAME')
            # conv_biases =
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

    def build(self, input):
        self.conv1 = tf.nn.relu(self.conv_layer(input, n_filters=64, name='conv1'))
        self.conv2 = tf.nn.relu(self.conv_layer(input, n_filters=64, name='conv2'))
        self.conv3 = tf.nn.relu(self.conv_layer(input, n_filters=64, name='conv3'))
        self.conv4 = tf.nn.softmax(self.conv_layer(input, n_filters=64, name='conv4'))
        self.output = self.conv4
