import cv2
import numpy as np
import progressbar
import os

import reader


def warp_img(img, flow):
    """ warp img following optical flow (image must be 1 channel)"""
    assert len(img.shape) == 2
    h, w = flow.shape[:2]
    i_, j_ = np.meshgrid(np.arange(w), np.arange(h))
    i_ = i_.reshape((h, w, 1))
    j_ = j_.reshape((h, w, 1))
    identity = np.concatenate((i_, j_), axis=2)
    map = (identity + flow).astype(np.float32)
    return cv2.remap(img, map1=map, map2=None, interpolation=cv2.INTER_LINEAR)


def warp_bgr(img, flow):
    """ warp img following optical flow """
    h, w = flow.shape[:2]
    i_, j_ = np.meshgrid(np.arange(w), np.arange(h))
    i_ = i_.reshape((h, w, 1))
    j_ = j_.reshape((h, w, 1))
    identity = np.concatenate((i_, j_), axis=2)
    map = (identity + flow).astype(np.float32)
    img_b = cv2.remap(img[:, :, 0], map1=map, map2=None, interpolation=cv2.INTER_LINEAR)
    img_g = cv2.remap(img[:, :, 1], map1=map, map2=None, interpolation=cv2.INTER_LINEAR)
    img_r = cv2.remap(img[:, :, 2], map1=map, map2=None, interpolation=cv2.INTER_LINEAR)
    img = np.concatenate((img_b.reshape((h, w, 1)), img_g.reshape((h, w, 1)), img_r.reshape((h, w, 1))), axis=2)
    return img


def correct_alpha(backward, forward, alpha):
    h, w = backward.shape[:2]
    err = np.zeros((h, w), dtype=np.float)
    # res = np.zeros()
    print(forward.shape)
    for i in range(h):
        for j in range(w):
            j0, i0 = backward[i, j]
            j0, i0 = min(int(j0 + j), w-1), min(int(i0 + i), h-1)
            # res[i, j] = img[i0, j0]
            j1, i1 = forward[i0, j0]
            j1, i1 = min(int(j1 + j0), w-1), min(int(i1 + i0), h-1)
            err[i, j] = np.linalg.norm(np.array([i1, j1] - np.array([i, j])))
    thresh = 15.
    alpha[np.where(err > thresh)] = 0.
    err_norm = cv2.normalize(err, None, 0., 1., cv2.NORM_MINMAX)
    cv2.imshow('Occlusion error', err_norm)
    # cv2.namedWindow('image')
    # def nothing(x):
    #     pass
    # cv2.createTrackbar('threshold', 'image', 0, 50, nothing)
    # while True:
    #     excl = np.zeros((h, w), dtype=np.float)
    #     thresh = cv2.getTrackbarPos('threshold', 'image')
    #     excl[np.where(err > thresh)] = 1.
    #     cv2.imshow('image', excl)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    return alpha


if __name__ == '__main__':
    flow_f = reader.read_flow('./test_data/forward.flo')
    flow_b = reader.read_flow('./test_data/backward.flo')
    alp, img = reader.read_fg_img('./test_data/in0062.png')
    # bgr = warp_bgr(img, flow_b)
    # bgr2 = warp_bgr(bgr, flow_f)
    h, w = img.shape[:2]
    alpha = warp_img(alp, flow_b)
    cv2.imshow('noncorr', alpha)
    alpha = correct_alpha(flow_b, flow_f, alpha)
    cv2.imshow('test', alpha)
    cv2.waitKey(0)
    # res = np.concatenate((bgr, (255.*alpha.reshape(img.shape[0], img.shape[1], 1)).astype(np.uint8)), axis=2) / 255.
    # res = bgr
    # vis = np.zeros((h, w, 3), dtype=np.float)
    # vis[:, :, 2] = err.astype(np.float)
    # vis = 0.5 * (vis + res)
    # cv2.imwrite('./test_data/res.png', res)
