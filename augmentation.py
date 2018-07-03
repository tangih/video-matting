import numpy as np
import cv2
import os
import progressbar

import reader
import tps


def object_size(alpha):
    """ typical side of foreground image """
    non_zero = np.zeros((alpha.shape[0], alpha.shape[1]), dtype=np.int)
    non_zero[np.where(alpha != 0.)] = 1
    return np.sqrt(np.sum(non_zero))


def fg_center(alpha):
    """ returns barycenter of fg image """
    non_zero = np.where(alpha != 0.)
    i, j = int(np.mean(non_zero[0])), int(np.mean(non_zero[1]))
    return j, i


def deform_grid(h, w, n=5):
    """ creates regular grid, applies random perturbation on it """
    fact = 0.05
    bound = min(w, h) * fact
    vec1 = (h / (n - 1)) * np.arange(n)
    vec2 = (w / (n - 1)) * np.arange(n)
    grid = np.transpose([np.repeat(vec1, n), np.tile(vec2, n)])
    new_grid = np.zeros_like(grid)
    for i in range(n * n):
        y = grid[i, 0]
        x = grid[i, 1]
        new_grid[i] = grid[i]
        if 0. < x < w:
            new_grid[i, 1] += np.random.uniform(-bound, bound)
        if 0. < y < h:
            new_grid[i, 0] += np.random.uniform(-bound, bound)
        # print('{} ----> {}'.format(grid[i], new_grid[i]))
    return grid, new_grid


def warp_image(img, params, thin=None):
    """ warp image according to given parameters """
    (tu, tv), rot, scale, center = params
    h, w = img.shape[:2]
    if thin is not None:
        grid, def_grid = thin
        if len(img.shape) == 3 and img.shape[2] == 3:
            def_img = tps.warp_images(grid, def_grid,
                                      [img[:, :, 0], img[:, :, 1], img[:, :, 2]],
                                      (0, 0, h, w), interpolation_order=1, approximate_grid=2)
            img = np.transpose(def_img, axes=(1, 2, 0)).copy()
        else:
            def_img = tps.warp_images(grid, def_grid, [img],
                                      (0, 0, h, w), interpolation_order=1, approximate_grid=2)[0]
            img = def_img
    mt = np.float32([[1, 0, tu], [0, 1, tv]])
    translated = cv2.warpAffine(img, mt, (w, h))
    mr = cv2.getRotationMatrix2D(center, rot, scale)
    rotated = cv2.warpAffine(translated, mr, (w, h))
    return rotated


def identity(m, n):
    """ returns array s.t. arr[i, j] = [i, j] """
    vec1 = np.arange(1, n+1)
    vec2 = np.arange(1, m+1)
    return np.transpose([np.repeat(vec2, n), np.tile(vec1, m)]).reshape(m, n, 2)


def synthetize_flow(fg_params, bg_params, grids, warped_alpha):
    """ compute optical flow associated to warp parameters """
    h, w = warped_alpha.shape[:2]
    id_fg = identity(h, w)
    id_bg = identity(h, w)
    w_fg = warp_image(id_fg, fg_params, thin=grids)
    w_bg = warp_image(id_bg, bg_params)
    # flow = np.zeros((h, w, 2), dtype=np.float)
    bi_alpha = np.zeros((h, w, 2), dtype=np.float)
    bi_alpha[:, :, 0] = warped_alpha
    bi_alpha[:, :, 1] = warped_alpha
    flow = np.multiply(bi_alpha, w_fg - id_fg) + np.multiply(1.-bi_alpha, w_bg-id_bg)
    return flow


def change_illumination(bgr, a, b, c):
    """ randomly changes illumination via [H]SV transformation """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    new_s = a * np.power(hsv[:, :, 1] / 255., b) + c
    new_v = a * np.power(hsv[:, :, 2] / 255., b) + c
    new_s = np.clip(new_s, 0., 1.)
    new_v = np.clip(new_v, 0., 1.)
    new_hsv = np.zeros_like(hsv)
    new_hsv[:, :, 0] = hsv[:, :, 0]
    new_hsv[:, :, 1] = (255. * new_s).astype(np.uint8)
    new_hsv[:, :, 2] = (255. * new_v).astype(np.uint8)
    return cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)


def augment(fg, bg, alpha):
    """ randomly modify input image to create synthetic data """
    bound_translate = 0.05
    bound_rotate = 10
    bound_scale = 0.15
    h, w = fg.shape[:2]
    fg_size = object_size(alpha)
    # camera motion (bg homography)
    tu_bg = int(np.random.uniform(-w * bound_translate, w * bound_translate))
    tv_bg = int(np.random.uniform(-h * bound_translate, h * bound_translate))
    rot_bg = 0.  # np.random.uniform(-bound_rotate, bound_rotate)
    scale_bg = np.random.uniform(1., 1. + bound_scale)
    params_bg = (tu_bg, tv_bg), rot_bg, scale_bg, (w//2, h//2)
    new_bg = warp_image(bg, params_bg)
    # object motion (fg homography + TPS deformation)
    grid, def_grid = deform_grid(h, w)
    tu_fg = int(np.random.uniform(-fg_size * bound_translate, fg_size * bound_translate))
    tv_fg = int(np.random.uniform(-fg_size * bound_translate, fg_size * bound_translate))
    rot_fg = np.random.uniform(-bound_rotate, bound_rotate)
    scale_fg = np.random.uniform(1., 1. + bound_scale)
    center = fg_center(alpha)
    params_fg = (tu_fg, tv_fg), rot_fg, scale_fg, center
    new_fg = warp_image(fg, params_fg, thin=(grid, def_grid))
    new_alpha = warp_image(alpha, params_fg, thin=(grid, def_grid))
    # modify illuminations parameters
    bound_a = 0.05
    bound_b = 0.3
    bound_c = 0.07
    a = np.random.uniform(1.-bound_a, 1.+bound_a)
    b = np.random.uniform(1.-bound_b, 1.+bound_b)
    c = np.random.uniform(-bound_c, bound_c)
    new_fg = change_illumination(new_fg, a, b, c)
    new_bg = change_illumination(new_bg, a, b, c)
    return new_fg, new_bg, new_alpha


def augmentation(dim_dataset, voc_dataset, sig_dataset):
    n = 50
    # we both take files from test and train datasets
    filepaths = [[os.path.join(dim_dataset, 'fg', folder, file)
                  for file in os.listdir(os.path.join(dim_dataset, 'fg', folder))]
                 for folder in ['DIM_TEST', 'DIM_TRAIN']]
    paths = filepaths[0] + filepaths[1]
    # we take VOC images as background
    voc_list = [os.path.join(voc_dataset, file) for file in os.listdir(voc_dataset)]
    dst_fg = os.path.join(sig_dataset, 'fg', 'augmented')
    dst_bg = os.path.join(sig_dataset, 'bg', 'augmented')
    for i, path in enumerate(paths):
        alpha, fg = reader.read_fg_img(path)
        name = os.path.basename(path).split('.')[0]
        print('Processing image {} ({}/{})'.format(name, i+1, len(paths)))
        bgra_ref = np.concatenate((fg, (255. * alpha.reshape((alpha.shape[0], alpha.shape[1], 1))).astype(np.uint8)),
                                  axis=2)
        cv2.imwrite(os.path.join(dst_fg, '{}_fg_ref.png'.format(name)), bgra_ref)
        for i in progressbar.progressbar(range(n)):
            bg_path = voc_list[np.random.randint(len(voc_list))]
            bg = cv2.imread(bg_path)
            bg = cv2.resize(bg, dsize=(fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)
            nfg, nbg, nal = augment(fg, bg, alpha)
            augmented_fg = np.concatenate((nfg, (255. * nal.reshape((nal.shape[0], nal.shape[1], 1))).astype(np.uint8)),
                                          axis=2)
            cv2.imwrite(os.path.join(dst_bg, '{}_bg_ref_{:04d}.png'.format(name, i)), bg)
            cv2.imwrite(os.path.join(dst_bg, '{}_bg_{:04d}.png'.format(name, i)), nbg)
            cv2.imwrite(os.path.join(dst_fg, '{}_fg_{:04d}.png'.format(name, i)), augmented_fg)


if __name__ == '__main__':
    # alpha, fg = reader.read_fg_img('test_data/in0062.png')
    # bg = cv2.imread('test_data/grass.jpg')
    # bg = cv2.resize(bg, dsize=(fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)
    # nfg, nbg, nal = augment(fg, bg, alpha)
    # cmp = reader.create_composite_image(fg, bg, alpha) / 255.
    # cmp2 = reader.create_composite_image(nfg, nbg, nal) / 255.
    # cv2.imshow('new', np.concatenate((cmp, cmp2), axis=0))
    # cv2.waitKey(0)
    datafolder = '/home/tangih/Documents/datasets/matting'
    augmentation(os.path.join(datafolder, 'DIM'), os.path.join(datafolder, 'VOC'), os.path.join(datafolder, 'SIG'))