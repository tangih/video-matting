import cv2
import numpy as np
import utils

VGG_MEAN = [103.939, 116.779, 123.68]


def read_fg_img(img_path):
    """ reads a foreground RGBA image """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:
        img = (((img+1) / 256.) - 1).astype(np.uint8)
        # avoid problems with uint16 images
    alpha = img[:, :, 3] / 255.
    bgr = img[:, :, :3]
    return alpha, bgr


def random_crop_and_pad(input, label, input_size):
    """ randomly crops image, if needed add zero-padding """
    crop_h, crop_w = input_size
    img_h, img_w = input.shape[:2]
    h, w = max(crop_h, img_h), max(crop_w, img_w)
    input_pad = np.zeros((h, w, 3), dtype=np.float)
    label_pad = np.zeros((h, w), dtype=np.float)
    input_pad[:h, :w, :] = input
    label_pad[:h, :w] = label
    i, j = np.random.randint(0, h-crop_h), np.random.randint(0, w-crop_w)
    input_crop = input_pad[i:i+crop_h, j:j+crop_w]
    label_crop = label_pad[i:i+crop_h, j:j+crop_w]
    return input_crop, label_crop


def random_scale(input, label):
    # TODO
    return [], []


def random_mirror(input, label):
    # TODO
    return [], []


def create_composite_image(fg, bg, alpha):
    """ creates composite """
    tri_alpha = np.zeros_like(fg, dtype=np.float)
    tri_alpha[:, :, 0] = alpha
    tri_alpha[:, :, 1] = alpha
    tri_alpha[:, :, 2] = alpha
    composite = np.multiply(tri_alpha, fg) + np.multiply(1. - tri_alpha, bg)
    return composite.astype(np.uint8)


def get_batch(file_list, input_size, img_mean, rd_scale=False, rd_mirror=False):
    """ returns normalized batch of cropped images (according to input_size)"""
    batch_size = len(file_list)
    input = np.zeros((batch_size, input_size[0], input_size[1], 3), dtype=np.float)
    label = np.zeros((batch_size, input_size[0], input_size[1]), dtype=np.float)
    for i in range(len(file_list)):
        fg_path, bg_path, tr_path = file_list[i]
        alpha, fg = read_fg_img(fg_path)
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        bg = cv2.imread(bg_path, 0)
        bg = bg.astype(dtype=np.float)
        fg = fg.astype(dtype=np.float)
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)
        cmp = np.multiply(alpha, fg) + np.multiply(1. - alpha, bg)
        cmp -= img_mean
        bg -= img_mean
        trimap = cv2.imread(tr_path, 0) / 255.
        trimap -= 0.5
        h, w = fg.shape[:2]
        inp = np.concatenate((cmp.reshape((h, w, 1)),
                              bg.reshape((h, w, 1)),
                              trimap.reshape((h, w, 1))), axis=2)
        if rd_scale:
            inp, alpha = random_scale(inp, alpha)
        if rd_mirror:
            inp, alpha = random_mirror(inp, alpha)
        input[i], label[i] = random_crop_and_pad(inp, alpha, input_size)
    return input, label

