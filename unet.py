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
    return tf.get_variable(initializer=conv, name='weights'), tf.get_variable(initializer=biases, name='WOOO')


class UNet:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(UNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'weights', 'vgg16.npy')
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def new_conv(self, input, n_filters, name):
        """ creates new convolution using Xavier initialisation """
        with tf.variable_scope(name):
            conv_weights, conv_biases = init_conv(3, 3, input.shape[-1], n_filters)
            conv = tf.nn.conv2d(input, conv_weights, [1, 1, 1, 1], padding='SAME')
            # conv_biases =
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def upconv_concat(self, input_a, input_b, n_filters, name):
        """Upsample `inputA` and concat with `input_B`

        Args:
            input_a (4-D Tensor): (N, H, W, 2*C)
            input_b (4-D Tensor): (N, 2*H, 2*H, C)
            n_filters: size of the up-conv layer
            name (str): name of the concat operation

        Returns:
            output (4-D Tensor): (N, 2*H, 2*W, 2*C)
        """
        with tf.variable_scope(name):
            h, w = input_b.shape[1:3]
            upsampled = tf.image.resize_images(input_a, [h, w])
            conv_weights, _ = init_conv(3, 3, upsampled.shape[-1], n_filters)
            up_conv = tf.nn.conv2d(upsampled, conv_weights, [1, 1, 1, 1], padding='SAME')

            return tf.concat(
                [up_conv, input_b], axis=-1, name="concat_{}".format(name))

    def conv_layer(self, bottom, name):
        """ creates convolution and loads associated VGG weights """
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            # print(filt.shape)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights")

    def get_conv_filter(self, name):
        pass


class UNetImage(UNet):
    def build(self, input):
        """
        load variable from npy to build the VGG

        :param input: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        self.conv1_1 = self.conv_layer(input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")

        # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        # self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        #
        # self.fc6 = self.fc_layer(self.pool5, "fc6")
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        # self.relu6 = tf.nn.relu(self.fc6)
        #
        # self.fc7 = self.fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)
        #
        # self.fc8 = self.fc_layer(self.relu7, "fc8")
        #
        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.upconv1 = self.upconv_concat(self.conv5_2, self.conv4_3, 512, name='upconv_1')
        self.conv4_4 = tf.nn.relu(self.new_conv(self.upconv1, 512, name='conv4_4'))
        # self.conv4_5 = tf.nn.relu(self.new_conv(self.conv4_4, 512, name='conv4_5'))
        self.upconv2 = self.upconv_concat(self.conv4_4, self.conv3_3, 256, name='upconv_2')
        self.conv3_4 = tf.nn.relu(self.new_conv(self.upconv2, 256, name='conv3_4'))
        # self.conv3_5 = tf.nn.relu(self.new_conv(self.conv3_4, 256, name='conv3_5'))
        self.upconv3 = self.upconv_concat(self.conv3_4, self.conv2_2, 128, name='upconv_3')
        self.conv2_3 = tf.nn.relu(self.new_conv(self.upconv3, 128, name='conv2_3'))
        # self.conv2_4 = tf.nn.relu(self.new_conv(self.conv2_3, 128, name='conv2_3'))
        self.upconv4 = self.upconv_concat(self.conv2_3, self.conv1_2, 64, name='upconv_4')
        # self.conv1_3 = tf.nn.relu(self.new_conv(self.upconv4, 64, name='conv1_3'))
        # self.conv1_4 = tf.nn.relu(self.new_conv(self.conv1_3, 64, name='conv1_4'))
        self.conv1_3 = self.new_conv(self.upconv4, 1, name='conv1_5')

        self.output = tf.nn.softmax(self.conv1_3, name='output')

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def get_conv_filter(self, name):
        if name == "conv1_1":
            old = self.data_dict[name][0]
            tens = np.zeros((3, 3, 7, 64), dtype=np.float32)
            tens[:, :, :3, :] = old
            tens[:, :, 3:6, :] = old
            return tf.Variable(tens, name='filter')
        return tf.Variable(self.data_dict[name][0], name="filter")


class UNetVideo(UNet):
    def build(self, input):
        """
        load variable from npy to build the VGG

        :param input: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        self.conv1_1 = self.conv_layer(input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")

        self.upconv1 = self.upconv_concat(self.conv5_2, self.conv4_3, 512, name='upconv_1')
        self.conv4_4 = tf.nn.relu(self.new_conv(self.upconv1, 512, name='conv4_4'))
        # self.conv4_5 = tf.nn.relu(self.new_conv(self.conv4_4, 512, name='conv4_5'))
        self.upconv2 = self.upconv_concat(self.conv4_4, self.conv3_3, 256, name='upconv_2')
        self.conv3_4 = tf.nn.relu(self.new_conv(self.upconv2, 256, name='conv3_4'))
        # self.conv3_5 = tf.nn.relu(self.new_conv(self.conv3_4, 256, name='conv3_5'))
        self.upconv3 = self.upconv_concat(self.conv3_4, self.conv2_2, 128, name='upconv_3')
        self.conv2_3 = tf.nn.relu(self.new_conv(self.upconv3, 128, name='conv2_3'))
        # self.conv2_4 = tf.nn.relu(self.new_conv(self.conv2_3, 128, name='conv2_3'))
        self.upconv4 = self.upconv_concat(self.conv2_3, self.conv1_2, 64, name='upconv_4')
        # self.conv1_3 = tf.nn.relu(self.new_conv(self.upconv4, 64, name='conv1_3'))
        # self.conv1_4 = tf.nn.relu(self.new_conv(self.conv1_3, 64, name='conv1_4'))
        self.conv1_3 = self.new_conv(self.upconv4, 1, name='conv1_5')

        self.output = tf.nn.softmax(self.conv1_3)

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def get_conv_filter(self, name):
        if name == "conv1_1":
            old = self.data_dict[name][0]
            tens = np.zeros((3, 3, 7, 64), dtype=np.float32)
            tens[:, :, :3, :] = old
            tens[:, :, 3:6, :] = old
            return tf.Variable(tens, name='filter')
        return tf.Variable(self.data_dict[name][0], name="filter")


if __name__ == '__main__':
    flags = {}
    model = UNetImage()
    x = tf.placeholder('float', [1, 321, 321, 7])
    model.build(x)
