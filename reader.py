import cv2
import numpy as np
import os


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


def load_test_image(filename='in0115.png', bg_name='sea.jpg'):
    """ loads a test image """
    path = os.path.join('test_data', filename)
    alpha, fg = read_fg_img(path)
    bg_path = os.path.join('test_data', bg_name)
    bg = cv2.imread(bg_path)
    h, w = fg.shape[:2]
    if bg.shape[0] != h or bg.shape[1] != w:
        bg = cv2.resize(bg, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
    cmp = create_composite_image(fg, bg, alpha)
    return fg, bg, cmp, alpha


def load_test_video(folder_name='hairball', bg_name='grass.jpg'):
    """ loads a test video """
    print('Loading test video...')
    image_names = sorted(os.listdir(os.path.join('test_data', folder_name)))
    h, w = cv2.imread(os.path.join('test_data', folder_name, image_names[0])).shape[:2]
    bg = cv2.imread(os.path.join('test_data', bg_name))
    if bg.shape[0] != h or bg.shape[1] != w:
        bg = cv2.resize(bg, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    fg_list = []
    alpha_list = []
    cmp_list = []
    for filename in image_names:

        alpha, fg = read_fg_img(os.path.join('test_data', folder_name, filename))
        cmp = create_composite_image(fg, bg, alpha).astype(np.uint8)
        fg_list.append(fg)
        alpha_list.append(alpha)
        cmp_list.append(cmp)
    return fg_list, alpha_list, cmp_list, bg


def vidshow(images, win_name='Video'):
    """ similar to cv2.imshow but for video """
    n_imgs = len(images)
    i = 0
    while True:
        cv2.imshow(win_name, images[i])
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        i = (i+1) % n_imgs


if __name__ == '__main__':
    vidshow(load_test_video()[2])
