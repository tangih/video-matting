import cv2
import numpy as np

import loader


def warp():
    pass


if __name__ == '__main__':
    al1, fg1 = loader.read_fg_img('./test_data/in0074.png')
    al2, fg2 = loader.read_fg_img('./test_data/in0075.png')
    bg = cv2.imread('./test_data/sea.jpg')
    prev = loader.create_composite_image(fg1, bg, al1).astype(np.uint8)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = loader.create_composite_image(fg2, bg, al2).astype(np.uint8)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    # TODO: test optical flow on synthetic examples
    flow = cv2.calcOpticalFlowFarneback(next_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # swap next/prev for warp
    # cv2.imshow('tst', cmp1)
    # cv2.waitKey(0)
    h, w = prev.shape[:2]
    identity = np.transpose(np.meshgrid(np.arange(h), np.arange(w)))
    map = identity + flow
    map_x = identity[:, :, 1] + flow[:, :, 0]
    map_y = identity[:, :, 0] + flow[:, :, 1]
    # map_y = map[:, :, 1]
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    res_b = cv2.remap(prev, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR)
    vis = np.zeros((prev.shape[0], prev.shape[1], 3), dtype=np.float)
    err = np.linalg.norm(res_b - next, axis=2) / 1000.
    print(np.max(err))
    cv2.imshow('test', err)
    cv2.waitKey(0)
    # cv2.imshow('test2',res)
    # cv2.imshow('test3',next_gray / 255.)

    cv2.waitKey(0)