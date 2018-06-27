import cv2
import numpy as np
import os

"""
READING
"""


def read_fg_img(img_path):
    """ reads a foreground RGBA image """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:
        img = (((img+1) / 256.) - 1).astype(np.uint8)
        # avoid problems with uint16 images
    alpha = img[:, :, 3] / 255.
    bgr = img[:, :, :3]
    return alpha, bgr


def read_flow(flow_path):
    """ read optical flow .exr file """
    with open(flow_path, 'rb') as f:
        key = np.fromfile(f, dtype=np.float32, count=1)
        if 202021.25 != key:
            print('ERROR: invalid key ({})'.format(key))
        w = np.fromfile(f, dtype=np.int32, count=1)[0]
        h = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=2*h*w).reshape((h, w, 2))
        return data


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


def load_test_video(folder_name='hairball2', bg_name='grass.jpg'):
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


"""
COMPOSITING
"""


def create_composite_image(fg, bg, alpha):
    """ creates composite """
    tri_alpha = np.zeros_like(fg, dtype=np.float)
    tri_alpha[:, :, 0] = alpha
    tri_alpha[:, :, 1] = alpha
    tri_alpha[:, :, 2] = alpha
    composite = np.multiply(tri_alpha, fg) + np.multiply(1. - tri_alpha, bg)
    return composite


def colorwheel(side, size):
    """ create a colorwheel """
    h, w = size
    center_i, center_j = int(0.75*side), int(0.75*side)
    orig_i, orig_j = int(0.25*side), int(0.25*side)
    flow = np.zeros((h, w, 2), dtype=np.float)
    alpha = np.zeros((h, w), dtype=np.float)
    for i in range(orig_i, orig_i+side):
        for j in range(orig_j, orig_j+side):
            s = np.linalg.norm(np.subtract([i, j], [center_i, center_j]))
            if s <= side/2:
                alpha[i, j] = 1.
                if s < 0.9*side/2:
                    flow[i, j, 1] = float(i - center_i)
                    flow[i, j, 0] = float(j - center_j)
    bgr = img_flow(flow)
    # cv2.imwrite('test.png', np.concatenate((bgr, (255.*alpha.reshape(h, w, 1)).astype(np.uint8)), axis=2))
    return bgr, alpha


def add_colorwheel(colorwheel, image):
    """ adds the colorwheel to the image """
    c_bgr, c_alpha = colorwheel
    cmp = create_composite_image(c_bgr, image, c_alpha)
    return cmp.astype(np.uint8)


"""
DISPLAYING
"""


def vidshow(images, vid_name='Video'):
    """ similar to cv2.imshow but for video """
    n_imgs = len(images)
    i = 0
    while True:
        cv2.imshow(vid_name, images[i])
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        i = (i+1) % n_imgs


def img_flow(flow):
    """ optical flow visualisation """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def vidflow_show(vid_img, vid_flow):
    """ show flow video """
    print('Creating visualisation')
    n_imgs = len(vid_img)
    assert len(vid_flow) == n_imgs
    step = 15
    vis_list = []
    h, w = vid_img[0].shape[:2]
    cw = colorwheel(150, (h, w))
    for k in range(n_imgs):
        arrows = vid_img[k]
        flow = vid_flow[k]
        for i in range(0, arrows.shape[0], step):
            for j in range(0, arrows.shape[1], step):
                cv2.arrowedLine(arrows, (j, i), (j+int(flow[i, j, 0]), i+int(flow[i, j, 1])), color=(0, 0, 255))
        flow_vis = img_flow(flow)
        flow_vis = add_colorwheel(cw, flow_vis)
        vis = np.concatenate((flow_vis, arrows), axis=0)
        vis_list.append(vis)
    vidshow(vis_list, vid_name='Optical flow visualisation')


if __name__ == '__main__':
    vidshow(load_test_video()[2])
