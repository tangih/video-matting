import cv2
import numpy as np
import os

import params



def read_fg_img(img_path):
    """ reads a foreground RGBA image """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:
        img = (((img+1) / 256.) - 1).astype(np.uint8)
        # avoid problems with uint16 images
    alpha = img[:, :, 3] / 255.
    bgr = img[:, :, :3]
    return alpha, bgr


def create_composite_image(fg, bg, alpha):
    """ creates composite """
    tri_alpha = np.zeros_like(fg, dtype=np.float)
    tri_alpha[:, :, 0] = alpha
    tri_alpha[:, :, 1] = alpha
    tri_alpha[:, :, 2] = alpha
    composite = np.multiply(tri_alpha, fg) + np.multiply(1. - tri_alpha, bg)
    return composite


def random_crop_and_pad(input, label, raw_fg=None, input_size=(320, 320)):
    """ randomly crops image, if needed add zero-padding """
    crop_type = [(320, 320), (480, 480), (640, 640)]  # we crop images of different sizes
    crop_h, crop_w = crop_type[np.random.randint(0, len(crop_type))]
    img_h, img_w = input.shape[:2]
    h, w = max(crop_h, img_h), max(crop_w, img_w)
    input_pad = np.zeros((h, w, 7), dtype=np.float)
    label_pad = np.zeros((h, w), dtype=np.float)
    input_pad[:img_h, :img_w, :] = input
    label_pad[:img_h, :img_w] = label
    # randomly picks top-left corner
    i, j = np.random.randint(0, h-crop_h+1), np.random.randint(0, w-crop_w+1)
    input_crop = input_pad[i:i+crop_h, j:j+crop_w]
    label_crop = label_pad[i:i+crop_h, j:j+crop_w]
    if input_size != (crop_h, crop_w):
        input_crop = cv2.resize(input_crop, input_size, interpolation=cv2.INTER_LINEAR)
        label_crop = cv2.resize(label_crop, input_size, interpolation=cv2.INTER_LINEAR)
    raw_fg_crop = None
    if raw_fg is not None:
        raw_fg_pad = np.zeros((h, w, 3), dtype=np.float)
        raw_fg_pad[:img_h, :img_w, :] = raw_fg
        raw_fg_crop = raw_fg_pad[i:i+crop_h, j:j+crop_w]
        if input_size != (crop_h, crop_w):
            raw_fg_crop = cv2.resize(raw_fg_crop, input_size, interpolation=cv2.INTER_LINEAR)
    return input_crop, label_crop, raw_fg_crop


def load_entry(entry):
    """ loads load input/label from training list entry """
    fg_path, tr_path, bg_path = entry
    alpha, fg = read_fg_img(fg_path)
    fg = fg.astype(dtype=np.float)
    bg = cv2.imread(bg_path).astype(dtype=np.float)
    raw_fg = fg[:, :, :3]
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)
    cmp = create_composite_image(fg, bg, alpha)
    trimap = cv2.imread(tr_path, 0) / 255.
    cmp -= params.VGG_MEAN
    bg -= params.VGG_MEAN
    trimap -= 0.5
    h, w = cmp.shape[:2]
    cv2.waitKey(0)
    inp = np.concatenate((cmp,
                          bg,
                          trimap.reshape((h, w, 1))), axis=2)
    return inp, alpha, raw_fg


def random_flip(input, label, raw_fg):
    """ mirrors image vertically with proba. 1/2 """
    if np.random.uniform(0., 1.) > 0.5:
        input = np.flip(input, axis=1)
        label = np.flip(label, axis=1)
        raw_fg = np.flip(raw_fg, axis=1)
    return input, label, raw_fg


def random_scale(input, label, raw_fg):
    # TODO
    return [], [], []


def get_batch(file_list, input_size, rd_scale=False, rd_mirror=False):
    """ returns normalized batch of cropped images (according to input_size)"""
    batch_size = len(file_list)
    input = np.zeros((batch_size, input_size[0], input_size[1], 3), dtype=np.float)
    label = np.zeros((batch_size, input_size[0], input_size[1]), dtype=np.float)
    raw_fgs = np.zeros((batch_size, input_size[0], input_size[1], 3), dtype=np.float)
    for i in range(len(file_list)):
        inp, lab, raw_fg = load_entry(file_list[i])
        if rd_scale:
            inp, alpha, raw_fg = random_scale(inp, lab, raw_fg)
        if rd_mirror:
            inp, alpha, raw_fg = random_flip(inp, lab, raw_fg)
        input[i], label[i], raw_fgs[i] = random_crop_and_pad(inp, lab, raw_fg, input_size)

    return input, label, raw_fgs


def show_entry(inp, lab, name):
    """ visualize training entry """
    assert inp.shape[0] == lab.shape[0] and inp.shape[1] == lab.shape[1]
    cmp = (inp[:, :, :3] + params.VGG_MEAN) / 255.
    bg = (inp[:, :, 3:6] + params.VGG_MEAN) / 255.
    trimap = inp[:, :, 6] + 0.5
    vis_trimap = np.repeat(trimap.reshape(trimap.shape[0], trimap.shape[1], 1), repeats=3, axis=2)
    vis_label = np.repeat(lab.reshape(lab.shape[0], lab.shape[1], 1), repeats=3, axis=2)
    row1 = np.concatenate((bg, cmp), axis=1)
    row2 = np.concatenate((vis_trimap, vis_label), axis=1)
    vis = np.concatenate((row1, row2), axis=0)
    cv2.imshow('Entry visualisation: {}'.format(name), vis)
    cv2.waitKey(0)


def get_file_list(root_dir, list_path):
    """ reads file list """
    files = []
    with open(list_path, 'r') as f:
        for line in f:
            fg_path, tr_path, bg_path = [os.path.join(root_dir, rel_path)
                                         for rel_path in line[:-1].split(' ')]
            files.append([fg_path, tr_path, bg_path])
    return files


def get_batch_list(file_list, batch_size):
    """ returns file list for current batch """
    batch_files = []
    for i in range(batch_size):
        batch_files.append(file_list.pop())
    return batch_files


def epoch_is_over(file_list, batch_size):
    """ returns true if there are not enough files in the training list to perform another iteration"""
    return len(file_list) < batch_size


if __name__ == '__main__':
    file_list = get_file_list(params.SYNTHETIC_DATASET, './dataset/valid.txt')
    entry = file_list[np.random.randint(0, len(file_list))]
    filename = entry[0].split('/')[-1]
    inp, lab, _ = load_entry(entry)
    inp, lab, _ = random_crop_and_pad(inp, lab, raw_fg=None, input_size=(320, 320))
    show_entry(inp, lab, name=filename)
