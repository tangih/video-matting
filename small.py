import tensorflow as tf
import numpy as np


def init_conv(win_h, win_w, n_inputs, n_filters):
    n = win_h * win_w * int(n_inputs)
    std = np.sqrt(2. / n)
    conv = np.random.normal(loc=0., scale=std, size=(win_h, win_w, n_inputs, n_filters)).astype(np.float32)
    biases = np.random.normal(loc=0., scale=std, size=n_filters).astype(np.float32)
    return tf.get_variable(initializer=conv, name='weights'), tf.get_variable(initializer=biases, name='biases')


def upconv_concat(prev_layer, downsampled, n_filters, name, phase):
    with tf.variable_scope(name):
        h, w = prev_layer.shape[:2]
        upsampled = tf.image.resize_images(downsampled, [h, w])
        conv_weights, _ = init_conv(3, 3, upsampled.shape[-1], n_filters)
        up_conv = tf.nn.relu(tf.nn.conv2d(upsampled, conv_weights, [1, 1, 1, 1], padding='SAME'))
        layers = [prev_layer, up_conv]

        return tf.contrib.layers.batch_norm(tf.concat(layers, axis=-1, name="concat_{}".format(name)),
                                            center=True, scale=True, is_training=phase, scope='bn')


def new_conv(input, n_filters, name, phase):
    """ creates new convolution using Xavier initialisation """
    with tf.variable_scope(name):
        conv_weights, conv_biases = init_conv(3, 3, input.shape[-1], n_filters)
        conv = tf.nn.conv2d(input, conv_weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        bn = tf.contrib.layers.batch_norm(bias, center=True, scale=True,
                                          is_training=phase, scope='bn')
        return bn


class UNetSmall:
    def __init__(self, input, phase):
        self.conv1_1 = tf.nn.relu(new_conv(input, 8, 'conv1_1', phase))
        self.pool1 = tf.nn.max_pool(self.conv1_1)
        self.conv2_1 = tf.nn.relu(new_conv(self.pool1, 16, 'conv2_1', phase))
        self.pool2 = tf.nn.max_pool(self.conv2_1)
        self.conv3_1 = tf.nn.relu(new_conv(self.pool2, 32, 'conv3_1', phase))
        self.conv3_2 = tf.nn.relu(new_conv(self.conv3_1, 32, 'conv3_2', phase))
        self.upconv1 = upconv_concat(self.conv2_1, self.conv3_2, 16, 'upconv1', phase)
        self.conv2_2 = tf.nn.relu(new_conv(self.upconv1, 16, 'conv2_2', phase))
        self.upconv2 = upconv_concat(self.conv1_1, self.conv2_2, 8, 'upconv2', phase)
        self.conv1_2 = tf.nn.relu(new_conv(self.upconv2, 8, 'conv1_2', phase))
        self.conv1_3 = new_conv(self.conv1_2, 1, 'conv1_2', phase)
        self.output = tf.nn.sigmoid(self.conv1_3, name='probs')
