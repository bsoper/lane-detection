import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
from scipy import misc
from warper import Warper


class Thresh:
    #def __init__(self):
        # self.thresh_dir_min = np.pi / 4
        # self.thresh_dir_max = np.pi / 2

        #self.thresh_mag_min = 150
        #self.thresh_mag_max = 255

    def thresh(self):
        img = cv2.imread('output_images/warped_color.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = img[:, :, 2]
        h = np.asarray(plt.hist(img.ravel(), range=[0, 255]))
        plt.close()

        # Gets all local minima and maxima
        maxi = np.array([h[0][scs.argrelmax(h[0])], h[1][scs.argrelmax(h[0])]])
        mini = np.array([h[0][scs.argrelmin(h[0])], h[1][scs.argrelmin(h[0])]])

        # finds max peak and two surrounding valleys and removes those pixels from the image
        l1 = np.argmax(mini[1] > maxi[1, np.argmax(maxi[0])])
        # print(np.array_str(mini[1,l1]))
        if l1 != 0:
            img[(mini[1, l1 - 1] < img) & (img < mini[1, l1])] = 0
        else:
            img[img < mini[1, l1]] = 0

        # Calculate scharr derivatives and adjust
        sx = np.absolute(cv2.Scharr(img, cv2.CV_64F, 1, 0))
        scale_factor = np.max(sx) / 255
        sx = (sx / scale_factor).astype(np.uint8)
        misc.imsave('output_images/thresh.jpg', sx)
        combined = np.zeros_like(sx)
        combined[sx > 0] = 1
        return combined