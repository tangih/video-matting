import cv2
import numpy as np

import loader


def warp():
    pass

def img_flow(flow):
    """ optical flow visualisation """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def colorwheel(side, size):
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
                if s < 0.95*side/2:
                    flow[i, j, 0] = float(i - center_i)
                    flow[i, j, 1] = float(j - center_j)
    bgr = img_flow(flow)
    return bgr, alpha


def add_colorwheel(colorwheel, image):
    c_bgr, c_alpha = colorwheel
    cmp = loader.create_composite_image(c_bgr, image, c_alpha)
    return cmp.astype(np.uint8)


def vidflow_show(vid_img, vid_flow):
    n_imgs = len(vid_img)
    assert len(vid_flow) == n_imgs
    step = 15
    vis_list = []
    h, w = vid_img[0].shape[:2]
    cw = colorwheel(150, (h, w))
    for k in range(n_imgs):
        arrows = vid_img[i]
        flow = vid_flow[i]
        for i in range(0, arrows.shape[0], step):
            for j in range(0, arrows.shape[1], step):
                cv2.arrowedLine(arrows, (j, i), (j+int(flow[i, j, 1]), i+int(flow[i, j, 0])), color=(0, 0, 255))
        flow_vis = img_flow(flow)
        vis = np.concatenate((flow_vis, arrows), axis=0)
        vis = add_colorwheel(colorwheel, vis)
        vis_list.append(vis)
    while True:
        for k in range(n_imgs):
            cv2.imshow('Optical flow visualisation', vis_list[i])
            cv2.waitKey(30)



# if __name__ == '__main__':
#     al1, fg1 = loader.read_fg_img('./test_data/in0115.png')
#     al2, fg2 = loader.read_fg_img('./test_data/in0119.png')
#     # bg = cv2.imread('./test_data/sea.jpg')
#     bg = np.zeros_like(fg1)
#
#     prev = loader.create_composite_image(fg1, bg, al1).astype(np.uint8)
#     prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#     next = loader.create_composite_image(fg2, bg, al2).astype(np.uint8)
#     next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
#     # TODO: test optical flow on synthetic examples
#     flow = cv2.calcOpticalFlowFarneback(next_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     # swap next/prev for warp
#     # cv2.imshow('tst', cmp1)
#     # cv2.waitKey(0)
#     h, w = prev.shape[:2]
#     step = 10
#     vis = np.mean(np.concatenate((prev.reshape(h, w, 3, 1), next.reshape(h, w, 3, 1)), axis=3),
#                   axis=3) / 255.
#     print(vis.shape)
#     for i in range(0, h, step):
#         for j in range(0, w, step):
#             cv2.arrowedLine(vis, (j, i), (int(j+flow[i, j, 0]), int(i+flow[i, j, 1])), color=(0, 0, 255))
#
#     # res_b = cv2.remap(prev, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR)
#     # vis = np.zeros((prev.shape[0], prev.shape[1], 3), dtype=np.float)
#     # err = np.linalg.norm(res_b - next, axis=2) / 700.
#     # print(np.mean(err))
#     # cv2.imshow('res', res_b)
#     # cv2.imshow('test', err)
#     cv2.imshow('test', vis)
#     cv2.waitKey(0)
#     # cv2.imshow('test2',res)
#     # cv2.imshow('test3',next_gray / 255.)
#     opts = {'DIS': cv2.optflow_DISOpticalFlow,
#             'PCA': cv2.optflow_OpticalFlowPCAFlow
#             }
#     for method in opts.keys():
#         opt = opts[method]
#         flow = np.zeros((h, w, 2), dtype=np.float)
#         opt.calc(prev, next, flow=flow)
#         vis = np.mean(np.concatenate((prev.reshape(h, w, 3, 1), next.reshape(h, w, 3, 1)), axis=3),
#                       axis=3) / 255.
#         for i in range(0, h, step):
#             for j in range(0, w, step):
#                 cv2.arrowedLine(vis, (j, i), (int(j + flow[i, j, 0]), int(i + flow[i, j, 1])), color=(0, 0, 255))
#         cv2.imshow('Optical flow, {} method'.format(method), vis)
#         cv2.waitKey(0)
#
#     # opt = cv2.optflow_DISOpticalFlow()
#     # flow = np.zeros_like()
#     # opt.calc(prev, next, flow=)
#     # opt.
#     # cv2.op
#     # cv2.waitKey(0)
