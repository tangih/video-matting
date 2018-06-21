import cv2
import numpy as np
import os

VGG_MEAN = [103.939, 116.779, 123.68]
DIM_TRAIN = '/home/tangih/Documents/datasets/image_matting/Adobe_Deep_Image_Matting_Dataset/Adobe-licensed/'
DIM_TEST = '/home/tangih/Documents/datasets/image_matting/Adobe_Deep_Image_Matting_Test_Set'
SYNTHETIC_DATASET = '/home/tangih/Documents/datasets/3d_models/SYNTHETIC/'
VOC_DATASET = '/home/tangih/Documents/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'


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
    input_pad = np.zeros((h, w, 7), dtype=np.float)
    label_pad = np.zeros((h, w), dtype=np.float)
    input_pad[:img_h, :img_w, :] = input
    label_pad[:img_h, :img_w] = label
    # randomly picks top-left corner
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
    return composite


def load_entry(entry):
    """ loads load input/label from training list entry """
    fg_path, tr_path, bg_path = entry
    alpha, fg = read_fg_img(fg_path)
    fg = fg.astype(dtype=np.float)
    bg = cv2.imread(bg_path).astype(dtype=np.float)
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)
    cmp = create_composite_image(fg, bg, alpha)
    trimap = cv2.imread(tr_path, 0) / 255.
    cmp -= VGG_MEAN
    bg -= VGG_MEAN
    trimap -= 0.5
    h, w = cmp.shape[:2]
    cv2.waitKey(0)
    inp = np.concatenate((cmp,
                          bg,
                          trimap.reshape((h, w, 1))), axis=2)
    return inp, alpha


def get_batch(file_list, input_size, rd_scale=False, rd_mirror=False):
    """ returns normalized batch of cropped images (according to input_size)"""
    batch_size = len(file_list)
    input = np.zeros((batch_size, input_size[0], input_size[1], 3), dtype=np.float)
    label = np.zeros((batch_size, input_size[0], input_size[1]), dtype=np.float)
    for i in range(len(file_list)):
        inp, lab = load_entry(file_list[i])
        if rd_scale:
            inp, alpha = random_scale(inp, lab)
        if rd_mirror:
            inp, alpha = random_mirror(inp, lab)
        input[i], label[i] = random_crop_and_pad(inp, lab, input_size)
    return input, label


def get_file_list(root_dir, list_path):
    """ reads file list """
    files = []
    with open(list_path, 'r') as f:
        for line in f:
            fg_path, tr_path, bg_path = [os.path.join(root_dir, rel_path)
                                         for rel_path in line[:-1].split(' ')]
            files.append([fg_path, tr_path, bg_path])
    return files


def show_entry(inp, lab, name):
    assert inp.shape[0] == lab.shape[0] and inp.shape[1] == lab.shape[1]
    cmp = (inp[:, :, :3] + VGG_MEAN) / 255.
    bg = (inp[:, :, 3:6] + VGG_MEAN) / 255.
    trimap = inp[:, :, 6] + 0.5
    vis_trimap = np.repeat(trimap.reshape(trimap.shape[0], trimap.shape[1], 1), repeats=3, axis=2)
    vis_label = np.repeat(lab.reshape(lab.shape[0], lab.shape[1], 1), repeats=3, axis=2)
    row1 = np.concatenate((bg, cmp), axis=1)
    row2 = np.concatenate((vis_trimap, vis_label), axis=1)
    vis = np.concatenate((row1, row2), axis=0)
    cv2.imshow('Entry visualisation: {}'.format(name), vis)
    cv2.waitKey(0)


if __name__ == '__main__':
    file_list = get_file_list(SYNTHETIC_DATASET, './dataset/train.txt')
    entry = file_list[np.random.randint(0, len(file_list))]
    filename = entry[0].split('/')[-1]
    inp, lab = load_entry(entry)
    inp, lab = random_crop_and_pad(inp, lab, (321, 321))
    show_entry(inp, lab, name=filename)