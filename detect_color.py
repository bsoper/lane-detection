import cv2
import numpy as np
from thresholder import Thresholder


def detect_color(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, pix_dim = img.shape

    left_side = hsv_img[:, 0:width / 2]
    right_side = hsv_img[:, width/2:]

    left_color = detect_single_side_color(left_side)
    right_color = detect_single_side_color(right_side)

    return left_color, right_color


def detect_single_side_color(hsv_img):
    thresholder = Thresholder()

    yellow = thresholder.yellow_thresh(hsv_img)
    white = thresholder.white_thresh(hsv_img)

    return ("White", "Yellow")[int(np.sum(white) < np.sum(yellow))]
