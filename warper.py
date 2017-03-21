import cv2
import numpy as np


class Warper:
    #def __init__(self):

    def set_transforms(self, video_size):
        ratio_x = video_size[0] / 1280
        ratio_y = video_size[1] / 720

        src = np.float32([
            [580 * ratio_x, 460 * ratio_y],
            [700 * ratio_x, 460 * ratio_y],
            [1040 * ratio_x, 680 * ratio_y],
            [260 * ratio_x, 680 * ratio_y],
        ])

        dst = np.float32([
            [260 * ratio_x, 0 * ratio_y],
            [1040 * ratio_x, 0 * ratio_y],
            [1040 * ratio_x, 720 * ratio_y],
            [260 * ratio_x, 720 * ratio_y],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img):
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, img):
        return cv2.warpPersective(
            img,
            self.Minv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )