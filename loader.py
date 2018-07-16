import cv2
import numpy as np
import os
import time

import params
import reader


def get_padded_img(img, crop_h, crop_w):
    """ returns padded image, the original image being randomly placed in the output window """
    h, w = img.shape[:2]
    new_h, new_w = max(crop_h, h), max(crop_w, w)
    padded_img = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    if crop_h > h:
        beg_i_out = np.random.randint(0, crop_h-h+1)
        end_i_out = beg_i_out + h
        beg_i_in = 0
        end_i_in = h
    else:
        beg_i_out = 0
        end_i_out = crop_h
        beg_i_in = np.random.randint(0, h-crop_h+1)
        end_i_in = beg_i_in + crop_h
    if crop_w > w:
        beg_j_out = np.random.randint(0, crop_w-w+1)
        end_j_out = beg_j_out + w
        beg_j_in = 0
        end_j_in = w
    else:
        beg_j_out = 0
        end_j_out = crop_w
        beg_j_in = np.random.randint(0, w-crop_w+1)
        end_j_in = beg_j_in + crop_w
    padded_img[beg_i_out:end_i_out, beg_j_out:end_j_out] = img[beg_i_in:end_i_in, beg_j_in:end_j_in]
    return padded_img


def load_and_crop(entry, input_size):
    """ loads load input/label from training list entry """
    fg_path, tr_path, bg_path = entry
    alpha, fg = reader.read_fg_img(fg_path)
    fg = fg.astype(dtype=np.float)  # potentially very big
    bg = cv2.imread(bg_path).astype(dtype=np.float)
    trimap = cv2.imread(tr_path, 0) / 255.
    trimap = trimap.reshape((trimap.shape[0], trimap.shape[1], 1))
    crop_type = [(320, 320), (480, 480), (640, 640)]  # we crop images of different sizes
    crop_h, crop_w = crop_type[np.random.randint(0, len(crop_type))]
    fg_h, fg_w = fg.shape[:2]
    if fg_h < crop_h or fg_w < crop_w:
        # in that case the image is not too big, and we have to add padding
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        cat = np.concatenate((fg, alpha, trimap), axis=2)
        cropped_cat = get_padded_img(cat, crop_h, crop_w)
        fg, alpha, trimap = np.split(cropped_cat, indices_or_sections=[3, 4], axis=2)
    # otherwise, the fg is likely to be HRes, we directly crop it and dismiss the original image
    # to avoid manipulation big images
    fg_h, fg_w = fg.shape[:2]
    i, j = np.random.randint(0, fg_h-crop_h+1), np.random.randint(0, fg_w-crop_w+1)
    fg = fg[i:i+crop_h, j:j+crop_h]
    alpha = alpha[i:i+crop_h, j:j+crop_h]
    trimap = trimap[i:i+crop_h, j:j+crop_h]
    # randomly picks top-left corner

    bg_crop_h, bg_crop_w = int(np.ceil(crop_h * bg.shape[0] / fg.shape[0])),\
                           int(np.ceil(crop_w * bg.shape[1] / fg.shape[1]))
    padded_bg = get_padded_img(bg, bg_crop_h, bg_crop_w)
    i, j = np.random.randint(0, bg.shape[0]-bg_crop_h+1), np.random.randint(0, bg.shape[1]-bg_crop_w+1)
    cropped_bg = padded_bg[i:i+bg_crop_h, j:j+bg_crop_w]
    bg = cv2.resize(src=cropped_bg, dsize=input_size, interpolation=cv2.INTER_LINEAR)
    fg = cv2.resize(fg, input_size, interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, input_size, interpolation=cv2.INTER_LINEAR)
    trimap = cv2.resize(trimap, input_size, interpolation=cv2.INTER_LINEAR)

    cmp = reader.create_composite_image(fg, bg, alpha)
    cmp -= params.VGG_MEAN
    bg -= params.VGG_MEAN
    trimap -= 0.5
    h, w = cmp.shape[:2]
    # inp = np.concatenate((cmp,
    #                       bg,
    #                       trimap.reshape((h, w, 1))), axis=2)
    inp = np.concatenate((cmp, bg), axis=2)
    label = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
    return inp, label, fg


def random_scale(input, label, raw_fg):
    # TODO
    return [], [], []


def get_batch(file_list, input_size, rd_scale=False, rd_mirror=False):
    """ returns normalized batch of cropped images (according to input_size)"""
    batch_size = len(file_list)
    input = np.zeros((batch_size, input_size[0], input_size[1], 6), dtype=np.float)
    label = np.zeros((batch_size, input_size[0], input_size[1], 1), dtype=np.float)
    raw_fgs = np.zeros((batch_size, input_size[0], input_size[1], 3), dtype=np.float)
    for i in range(len(file_list)):
        ref = time.time()
        # print(file_list[i])
        inp, lab, raw_fg = load_and_crop(file_list[i], input_size)
        if rd_scale:
            inp, lab, raw_fg = random_scale(inp, lab, raw_fg)
        if rd_mirror:
            if np.random.uniform(0., 1.) > 0.5:
                inp = np.flip(inp, axis=1)
                lab = np.flip(lab, axis=1)
                raw_fg = np.flip(raw_fg, axis=1)

        input[i], label[i], raw_fgs[i] = inp, lab, raw_fg
        # print('{} loaded in {:.3f}s (shape: {}x{})'.format(file_list[i][0].split('/')[-1],
        #                                                    time.time()-ref,
        #                                                    inp.shape[0],
        #                                                    inp.shape[1]))
    return input, label, raw_fgs


def show_entry(inp, lab, name):
    """ visualize training entry """
    assert inp.shape[0] == lab.shape[0] and inp.shape[1] == lab.shape[1]
    cmp = (inp[:, :, :3] + params.VGG_MEAN) / 255.
    bg = (inp[:, :, 3:6] + params.VGG_MEAN) / 255.
    # trimap = inp[:, :, 6] + 0.5
    # vis_trimap = np.repeat(trimap.reshape(trimap.shape[0], trimap.shape[1], 1), repeats=3, axis=2)
    vis_label = np.repeat(lab.reshape(lab.shape[0], lab.shape[1], 1), repeats=3, axis=2)
    row1 = np.concatenate((bg, cmp), axis=1)
    # row2 = np.concatenate((vis_trimap, vis_label), axis=1)
    row2 = np.concatenate((np.zeros((vis_label.shape[0], vis_label.shape[1], 3)), vis_label), axis=1)
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


def psnr(img, img_ref):
    """
    returns peak signal to noise ratio
    images have to be [0, 1] float
    """
    img_ = img.copy()
    img_ref_ = img_ref.copy()
    if len(img.shape) == 2:
        img_ = img_.reshape((img.shape[0], img.shape[1], 1))
    if len(img_ref.shape) == 2:
        img_ref_ = img_ref_.reshape((img_ref.shape[0], img_ref.shape[1], 1))
    eqm = np.mean(np.square(np.linalg.norm(img_ - img_ref_, axis=2)))
    eps = 1e-6  # to avoid dividing by zero
    return 10. * np.log10(1./(eps+eqm))


def add_noise(img, var=0.1):
    noised = img.copy()
    if len(img.shape) == 2:
        noised = noised.reshape((img.shape[0], img.shape[1], 1))
    h, w, n = noised.shape
    noise = np.random.normal(0., var, (h, w, n))
    noised += noise
    return np.clip(noised, 0, 1)


if __name__ == '__main__':
    # file_list = get_file_list(params.SYNTHETIC_DATASET, './dataset/valid.txt')
    # for i in range(10):
    #     inp, lab, fg = get_batch([file_list[np.random.randint(0, len(file_list))]], (320, 320), False, False)
    #     show_entry(inp[0], lab[0], name='test')
    _, _, img, alpha = reader.load_test_image()
    img /= 255.
    var = 0.1
    noised = add_noise(img, var=var)
    noised_al = add_noise(alpha, var=var)
    print('PSNR between the images is {:.5f}'.format(psnr(noised, img)))
    print('PSNR between the images is {:.5f}'.format(psnr(noised_al, alpha)))
    cv2.imshow('adding noise', np.concatenate((img, noised), axis=0))
    cv2.imshow('adding noise to alpha', np.concatenate((alpha, noised_al.reshape((alpha.shape[0], alpha.shape[1]))), axis=0))
    cv2.waitKey(0)
