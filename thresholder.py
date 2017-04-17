import cv2
import numpy as np


class Thresholder:
    def __init__(self):
        self.sobel_kernel = 15

        self.thresh_dir_min = 0.7
        self.thresh_dir_max = 1.2

        self.thresh_mag_min = 50
        self.thresh_mag_max = 255

    def dir_thresh(self, sobelx, sobely):
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.thresh_dir_min) & (scaled_sobel <= self.thresh_dir_max)] = 1

        return sxbinary

    def mag_thresh(self, sobelx, sobely):
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self.thresh_mag_min) & (gradmag <= self.thresh_mag_max)] = 1

        return binary_output

    def yellow_thresh(self, hsv_img):
        yellow_min = np.array([15, 100, 120], np.uint8)
        yellow_max = np.array([80, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsv_img, yellow_min, yellow_max)

        binary_output = np.zeros_like(hsv_img[:, :, 0])
        binary_output[yellow_mask != 0] = 1

        return binary_output

    def white_thresh(self, hsv_img):
        white_min = np.array([0, 0, 200], np.uint8)
        white_max = np.array([255, 30, 255], np.uint8)
        white_mask = cv2.inRange(hsv_img, white_min, white_max)

        binary_output = np.zeros_like(hsv_img[:, :, 0])
        binary_output[white_mask != 0] = 1

        return binary_output

    def color_thresh(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        base = self.yellow_thresh(img)
        white = self.white_thresh(img)

        base[white == 1] = 1

        return base

    def threshold(self, img, use_sobel=False):
        sobelx = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)

        direc = self.dir_thresh(sobelx, sobely)
        mag = self.mag_thresh(sobelx, sobely)
        color = self.color_thresh(img)

        combined = np.zeros_like(direc)
        #combined[((color == 1) & ((mag == 1) | (direc == 1)))] = 1
        if use_sobel:
            combined[(mag == 1)] = 1
        else:
            combined[(color == 1)] = 1

        return combined
