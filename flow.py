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


if __name__ == '__main__':
    pass
