import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane
from scipy import misc
import sys
from random import randint


def detect_dotted(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    left_side = img[:, 0:width / 2]
    right_side = img[:, width / 2:]

    left_dotted = detect_dotted_side(left_side)
    right_dotted = detect_dotted_side(right_side)

    return left_dotted, right_dotted


def detect_dotted_side(img):
    height, width = img.shape
    row_sum = np.sum(img, axis=1)
    top_img_nonzero = np.where(row_sum > 100)[0][0]
    num_zero = 0

    for i in range(100):
        index = randint(top_img_nonzero,height-1)
        trimmed = np.trim_zeros(img[index,:])
        trimmed = trimmed[np.where(trimmed > 50)]
        lane_width = trimmed.shape[0]
        if lane_width == 0:
            num_zero += 1
    return ("Dash", "Solid")[int(num_zero < 25)]

if __name__ == '__main__':
    left = detect_dotted(sys.argv[1])
    # plt.imshow(detect_dotted('warped3.jpg'),cmap = 'gray')
    # plt.show()
