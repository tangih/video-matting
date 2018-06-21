import tensorflow as tf
import numpy as np
import cv2

import loader
import unet

def train(train_list, val_list, n_epochs):
    model = unet.UNet()
    x = tf.placeholder('float', [None, None, None, 7])
    y = tf.placeholder('float', [None, None, None])
    model.build(x)