import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane


def detect_color(img):
    #img = cv2.imread(img, cv2.IMREAD_COLOR)
    height, width, pix_dim = img.shape

    img_all = img.reshape(-1,3)
    img_colors = img_all[np.sum(img_all, axis=1) > 100]
    img_averages = np.mean(img_colors, axis=0)
    img_norm = np.sum(np.square(img_averages - img_averages.mean()))
    color = ("White", "Yellow")[int(img_norm > 1000)]

    return color

