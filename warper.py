import cv2
import numpy as np
import matplotlib.pyplot as plt


class Warper:
    def __init__(self):
        self.src_n = np.array([])
        self.ratio = np.array([])
        self.dst = np.array([])
        self.src_o = np.array([])
        self.trap_area = 0.
        self.change = False
        self.M = np.array([])
        self.Minv = np.array([])

    def check_src_n(self):
        margin = 10 * self.ratio[0, 0]
        # Check and adjust trapezoid if lane change to the right is occuring
        if self.src_n[2, 0] < int(self.ratio[0, 0] * 640):
            temp = self.src_n[2, 0]
            self.src_n[2, 0] = self.ratio[0, 0] * 1280 - self.src_n[3, 0] + margin
            self.src_n[3, 0] = temp - margin
            temp = self.src_n[1, 0]
            self.src_n[1, 0] = self.ratio[0, 0] * 1280 - self.src_n[0, 0] + margin
            self.src_n[0, 0] = temp - margin
            print('Lane change to right')
            print(np.array_str(self.src_n))
            self.change = True

        # Check and adjust trapezoid if lane change to the left is occuring
        elif self.src_n[3, 0] > int(self.ratio[0, 0] * 640):
            temp = self.src_n[3, 0]
            self.src_n[3, 0] = self.ratio[0, 0] * 1280 - self.src_n[2, 0] - margin
            self.src_n[2, 0] = temp + margin
            temp = self.src_n[0, 0]
            self.src_n[0, 0] = self.ratio[0, 0] * 1280 - self.src_n[1, 0] - margin
            self.src_n[1, 0] = temp + margin
            print('Lane change to left')
            print(np.array_str(self.src_n))
            self.change = True

    def set_transforms(self, img):
        self.ratio = np.float32([[img.shape[1] / 1280, 0], [0, img.shape[0] / 720]])
        self.dst = np.int16([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])
        self.dst = np.int16(np.matmul(self.dst, self.ratio))
        # Check if src_n points are available, if not => use GUI to select src_n points
        if self.src_n.size:

            # Check if any changes have been made to stored points
            if (self.src_o != self.src_n).any():
                self.src_n = np.int16(self.src_n)
                self.check_src_n()
                self.src_o = np.copy(self.src_n)
                self.M = cv2.getPerspectiveTransform(np.float32(self.src_n), np.float32(self.dst))
                self.Minv = cv2.getPerspectiveTransform(np.float32(self.dst), np.float32(self.src_n))

        else:
            # Show image and expected point locations and ask for user input to get new src_n points
            plt.imshow(img)
            self.src_o = np.int16([
                [580, 460],
                [700, 460],
                [1040, 680],
                [260, 680],
            ])
            self.src_o = np.int16(np.matmul(self.src_o, self.ratio))
            plt.scatter(self.src_o[:, 0], self.src_o[:, 1], c='b', s=100, alpha=0.3)
            plt.title(
                'Select FOUR best trapezoid points \n Starting from TOP-LEFT corner in CLOCKWISE manner \n' +
                ' (RIGHT CLICK to cancel last selected point)')
            self.src_n = plt.ginput(4)
            plt.close()
            # operate on selected points to make them usable
            self.src_n = np.asarray(self.src_n, dtype=np.int16)
            self.src_n[0:2, 1] = (self.src_n[0, 1] + self.src_n[1, 1]) / 2
            self.src_n[2:4, 1] = (self.src_n[2, 1] + self.src_n[3, 1]) / 2
            self.src_n = np.int16(self.src_n)
            # Calculate transforms and save the src_n points
            self.M = cv2.getPerspectiveTransform(np.float32(self.src_n), np.float32(self.dst))
            self.Minv = cv2.getPerspectiveTransform(np.float32(self.dst), np.float32(self.src_n))
            self.src_o = np.copy(self.src_n)
            self.trap_area = ((self.src_n[1, 0] - self.src_n[0, 0]) + (self.src_n[2, 0] - self.src_n[3, 0])) \
                             / 2.0 * (self.src_n[2, 1] - self.src_n[1, 1])

    def warp(self, img):
        self.set_transforms(img)
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, img):
        return cv2.warpPerspective(
            img,
            self.Minv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )