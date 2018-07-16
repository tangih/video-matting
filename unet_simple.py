import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


def init_conv(win_h, win_w, n_inputs, n_filters):
    """ creates initial tensor using Xavier method """
    n = win_h * win_w * int(n_inputs)
    std = np.sqrt(2. / n)
    conv = np.random.normal(loc=0., scale=std, size=(win_h, win_w, n_inputs, n_filters)).astype(np.float32)
    biases = np.random.normal(loc=0., scale=std, size=n_filters).astype(np.float32)
    return tf.get_variable(initializer=conv, name='weights'), tf.get_variable(initializer=biases, name='biases')


def new_conv(input, n_filters, name):
    """ creates new convolution using Xavier initialisation """
    with tf.variable_scope(name):
        conv_weights, conv_biases = init_conv(3, 3, input.shape[-1], n_filters)
        conv = tf.nn.conv2d(input, conv_weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)
        return relu


def upconv_concat(prev_layers, prev, n_filters, name):
    with tf.variable_scope(name):
        h, w = prev_layers[0].shape[1:3]
        upsampled = tf.image.resize_images(prev, [h, w])
        conv_weights, _ = init_conv(3, 3, upsampled.shape[-1], n_filters)
        up_conv = tf.nn.relu(tf.nn.conv2d(upsampled, conv_weights, [1, 1, 1, 1], padding='SAME'))
        layers = []
        for layer in prev_layers:
            layers.append(layer)
        layers.append(up_conv)

        return tf.concat(layers, axis=-1, name="concat_{}".format(name))


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = tf.nn.relu(self.conv_layer(bgr, "conv1_1"))
        self.conv1_2_l = self.conv_layer(self.conv1_1, "conv1_2")
        self.conv1_2 = tf.nn.relu(self.conv1_2_l)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = tf.nn.relu(self.conv_layer(self.pool1, "conv2_1"))
        self.conv2_2_l = self.conv_layer(self.conv2_1, "conv2_2")
        self.conv2_2 = tf.nn.relu(self.conv2_2_l)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = tf.nn.relu(self.conv_layer(self.pool2, "conv3_1"))
        self.conv3_2 = tf.nn.relu(self.conv_layer(self.conv3_1, "conv3_2"))
        self.conv3_3_l = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_3 = tf.nn.relu(self.conv3_3_l)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = tf.nn.relu(self.conv_layer(self.pool3, "conv4_1"))
        self.conv4_2 = tf.nn.relu(self.conv_layer(self.conv4_1, "conv4_2"))
        self.conv4_3_l = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_3 = tf.nn.relu(self.conv4_3_l)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = tf.nn.relu(self.conv_layer(self.pool4, "conv5_1"))
        self.conv5_2 = tf.nn.relu(self.conv_layer(self.conv5_1, "conv5_2"))
        self.conv5_3_l = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_3 = tf.nn.relu(self.conv5_3_l)
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', trainable=False)

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases, trainable=False)

            return bias

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


class UNetSimple:
    def __init__(self, layers):
        self.layers = layers
        self.select4_1 = new_conv(layers['conv4'][0], 16, name='select4_1')
        self.select4_2 = new_conv(layers['conv4'][1], 16, name='select4_2')
        self.select4_3 = new_conv(layers['conv4'][2], 16, name='select4_3')
        self.upconv4 = upconv_concat([self.select4_1, self.select4_2, self.select4_3],
                                     layers['conv5'][-1], 48, name='upconv4')
        self.conv4 = new_conv(self.upconv4, 48, name='conv4')
        self.select3_1 = new_conv(layers['conv3'][0], 8, name='select3_1')
        self.select3_2 = new_conv(layers['conv3'][1], 8, name='select3_2')
        self.select3_3 = new_conv(layers['conv3'][2], 8, name='select3_3')
        self.upconv3 = upconv_concat([self.select3_1, self.select3_2, self.select3_3],
                                     self.conv4, 24, name='upconv3')
        self.conv3 = new_conv(self.upconv3, 24, name='conv3')
        self.select2_1 = new_conv(layers['conv2'][0], 4, name='select2_1')
        self.select2_2 = new_conv(layers['conv2'][1], 4, name='select2_2')
        self.upconv2 = upconv_concat([self.select2_1, self.select2_2],
                                     self.conv3, 24, name='upconv2')
        self.conv2 = new_conv(self.upconv2, 32, name='conv2')
        self.select1_1 = new_conv(layers['conv1'][0], 2, name='select1_1')
        self.select1_2 = new_conv(layers['conv1'][1], 2, name='select1_2')
        self.select1_3 = new_conv(layers['conv1'][2], 2, name='select1_3')
        self.upconv1 = upconv_concat([self.select1_1, self.select1_2, self.select1_3],
                                     self.conv2, 24, name='upconv1')
        self.conv1 = new_conv(self.upconv1, 32, name='conv1')
        self.output = new_conv(self.conv1, 1, name='output')


def create_model(cmp, bg, diff):
    vgg1 = Vgg16()
    vgg2 = Vgg16()
    vgg3 = Vgg16()
    vgg1.build(cmp)
    vgg2.build(bg)
    vgg3.build(diff)
    with tf.name_scope('feature_extract'):
        layers = {
            'conv1': [tf.concat([cmp, bg, diff], axis=-1, name='input'),
                      tf.concat([vgg1.conv1_1, vgg2.conv1_1, vgg3.conv1_1], axis=-1, name='conv1_1'),
                      tf.concat([vgg1.conv1_2, vgg2.conv1_2, vgg3.conv1_2], axis=-1, name='conv1_2')],
            'conv2': [tf.concat([vgg1.conv2_1, vgg2.conv2_1, vgg3.conv2_1], axis=-1, name='conv2_1'),
                      tf.concat([vgg1.conv2_2, vgg2.conv2_2, vgg3.conv2_2], axis=-1, name='conv2_2')],
            'conv3': [tf.concat([vgg1.conv3_1, vgg2.conv3_1, vgg3.conv3_1], axis=-1, name='conv3_1'),
                      tf.concat([vgg1.conv3_2, vgg2.conv3_2, vgg3.conv3_2], axis=-1, name='conv3_2'),
                      tf.concat([vgg1.conv3_3, vgg2.conv3_3, vgg3.conv3_3], axis=-1, name='conv3_3')],
            'conv4': [tf.concat([vgg1.conv4_1, vgg2.conv4_1, vgg3.conv4_1], axis=-1, name='conv4_1'),
                      tf.concat([vgg1.conv4_2, vgg2.conv4_2, vgg3.conv4_2], axis=-1, name='conv4_2'),
                      tf.concat([vgg1.conv4_3, vgg2.conv4_3, vgg3.conv4_3], axis=-1, name='conv4_3')],
            'conv5': [tf.concat([vgg1.conv5_1, vgg2.conv5_1, vgg3.conv5_1], axis=-1, name='conv5_1'),
                      tf.concat([vgg1.conv5_2, vgg2.conv5_2, vgg3.conv5_2], axis=-1, name='conv5_2'),
                      tf.concat([vgg1.conv5_3, vgg2.conv5_3, vgg3.conv5_3], axis=-1, name='conv5_3')]
        }
    model = UNetSimple(layers)
    return model

