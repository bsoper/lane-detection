import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane


def detect_color(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    height, width, pix_dim = img.shape

    left_side = img[:, 0:width / 2]
    right_side = img[:, width/2:]

    left_all = left_side.reshape(-1, 3)
    left_colors = left_all[np.sum(left_all, axis=1) > 100]
    left_averages = np.mean(left_colors, axis=0)

    right_all = right_side.reshape(-1, 3)
    right_colors = right_all[np.sum(right_all, axis=1) > 100]
    right_averages = np.mean(right_colors, axis=0)

    left_norm = np.sum(np.square(left_averages - left_averages.mean()))
    right_norm = np.sum(np.square(right_averages - right_averages.mean()))

    left_color = ("white", "yellow")[int(left_norm > 1000)]
    right_color = ("white", "yellow")[int(right_norm > 1000)]

    return left_color, right_color

