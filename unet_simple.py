import inspect
import os

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


def init_conv(win_h, win_w, n_inputs, n_filters):
    """ creates initial tensor using Xavier method """
    n = win_h * win_w * int(n_inputs)
    std = np.sqrt(2. / n)
    conv = np.random.normal(loc=0., scale=std, size=(win_h, win_w, n_inputs, n_filters)).astype(np.float32)
    biases = np.random.normal(loc=0., scale=std, size=n_filters).astype(np.float32)
    return tf.get_variable(initializer=conv, name='weights'), tf.get_variable(initializer=biases, name='biases')


def new_conv(input, n_filters, name, phase):
    """ creates new convolution using Xavier initialisation """
    with tf.variable_scope(name):
        conv_weights, conv_biases = init_conv(3, 3, input.shape[-1], n_filters)
        conv = tf.nn.conv2d(input, conv_weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        bn = tf.contrib.layers.batch_norm(bias, center=True, scale=True,
                                          is_training=phase, scope='bn')
        return bn


def upconv_concat(prev_layers, prev, n_filters, name, phase):
    with tf.variable_scope(name):
        h, w = prev_layers[0].shape[1:3]
        upsampled = tf.image.resize_images(prev, [h, w])
        conv_weights, _ = init_conv(3, 3, upsampled.shape[-1], n_filters)
        up_conv = tf.nn.relu(tf.nn.conv2d(upsampled, conv_weights, [1, 1, 1, 1], padding='SAME'))
        layers = []
        for layer in prev_layers:
            layers.append(layer)
        layers.append(up_conv)

        return tf.contrib.layers.batch_norm(tf.concat(layers, axis=-1, name="concat_{}".format(name)),
                                            center=True, scale=True, is_training=phase, scope = 'bn')


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'weights', 'vgg16.npy')
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, bgr):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

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


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            return bias

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")


class UNetSimple:
    def __init__(self, layers, phase):
        self.layers = layers
        self.select4_1 = tf.nn.relu(new_conv(layers['conv4'][0], 16, 'select4_1', phase))
        self.select4_2 = tf.nn.relu(new_conv(layers['conv4'][1], 16, 'select4_2', phase))
        self.select4_3 = tf.nn.relu(new_conv(layers['conv4'][2], 16, 'select4_3', phase))
        self.upconv4 = upconv_concat([self.select4_1, self.select4_2, self.select4_3],
                                     layers['conv5'][-1], 48, 'upconv4', phase)
        self.conv4 = tf.nn.relu(new_conv(self.upconv4, 48, 'conv4', phase))
        self.select3_1 = tf.nn.relu(new_conv(layers['conv3'][0], 8, 'select3_1', phase))
        self.select3_2 = tf.nn.relu(new_conv(layers['conv3'][1], 8, 'select3_2', phase))
        self.select3_3 = tf.nn.relu(new_conv(layers['conv3'][2], 8, 'select3_3', phase))
        self.upconv3 = upconv_concat([self.select3_1, self.select3_2, self.select3_3],
                                     self.conv4, 24, 'upconv3', phase)
        self.conv3 = tf.nn.relu(new_conv(self.upconv3, 24, 'conv3', phase), name='conv3')
        self.select2_1 = tf.nn.relu(new_conv(layers['conv2'][0], 4, 'select2_1', phase))
        self.select2_2 = tf.nn.relu(new_conv(layers['conv2'][1], 4, 'select2_2', phase))
        self.upconv2 = upconv_concat([self.select2_1, self.select2_2],
                                     self.conv3, 24, 'upconv2', phase)
        self.conv2 = tf.nn.relu(new_conv(self.upconv2, 32, 'conv2', phase), name='conv2')
        self.select1_1 = tf.nn.relu(new_conv(layers['conv1'][0], 2, 'select1_1', phase))
        self.select1_2 = tf.nn.relu(new_conv(layers['conv1'][1], 2, 'select1_2', phase))
        self.select1_3 = tf.nn.relu(new_conv(layers['conv1'][2], 2, 'select1_3', phase))
        self.upconv1 = upconv_concat([self.select1_1, self.select1_2, self.select1_3],
                                     self.conv2, 24, 'upconv1', phase)
        self.conv1 = tf.nn.relu(new_conv(self.upconv1, 32, 'conv1', phase), name='conv1')
        self.output = tf.nn.sigmoid(new_conv(self.conv1, 1, 'output', phase), name='probs')


def create_model(cmp, bg, diff, phase):
    vgg1 = Vgg16()
    vgg2 = Vgg16()
    vgg3 = Vgg16()
    vgg1.build(cmp)
    vgg2.build(bg)
    vgg3.build(diff)
    with tf.variable_scope('feature_extract'):
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
    with tf.variable_scope('simple_unet'):
        model = UNetSimple(layers, phase)
    return model
