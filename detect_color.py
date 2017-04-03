import cv2
import numpy as np
from matplotlib import pyplot as plt

from thresholder import Thresholder
import imutils
from lane import Lane


def detect_color(img):
    thresholder = Thresholder()
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, pix_dim = img.shape

    left_side = img[:, 0:width / 2]
    right_side = img[:, width/2:]

    yellow_left = thresholder.yellow_thresh(left_side)
    white_left = thresholder.white_thresh(hsv_img)

    yellow_right = thresholder.yellow_thresh(right_side)
    white_right = thresholder.white_thresh(right_side)

    left_color = ("White", "Yellow")[int(np.sum(white_left) < np.sum(yellow_left))]
    right_color = ("White", "Yellow")[int(np.sum(white_right) < np.sum(yellow_right))]

    return left_color, right_color

