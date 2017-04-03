import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane
from scipy import misc
import sys
from random import randint


def detect_dotted(img):
    #img = cv2.imread(img,0)
    height, width = img.shape
    row_sum = np.sum(img, axis=1)
    top_img_nonzero = np.nonzero(row_sum)[0][0]
    num_zero = 0

    for i in range(100):
      index = randint(top_img_nonzero,height-1)
      trimmed = np.trim_zeros(img[index,:])
      trimmed = trimmed[np.where(trimmed > 50)]
      lane_width = trimmed.shape[0]
      if lane_width == 0:
        num_zero += 1

    return ("Dash", "Solid")[int(num_zero < 25)]






    # median = cv2.medianBlur(img,11)
    # edges = cv2.Canny(median,10,20)
    # # im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(median, contours, 0, (128,255,0), 3)
    # cnts = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # # loop over the contours
    # height, width = img.shape
    # count = 0

    # for c in cnts:
    #   # compute the center of the contour
    #   M = cv2.moments(c)
    #   area = int(M['m00'])
    #   if area == 0:
    #     continue
    #   cX = int(M["m10"] / M["m00"])
    #   cY = int(M["m01"] / M["m00"])
    #   # draw the contour and center of the shape on the image

    #   if area > 2000:
    #       count += 1

    # if count > 1:
    #   solid_dash = 'Dash'
    # elif count == 1:
    #   solid_dash = 'Solid'
    # else:
    #   solid_dash = -1
    
    # return solid_dash
         

if __name__ == '__main__':
    left = detect_dotted(sys.argv[1])
    #plt.imshow(detect_dotted('warped3.jpg'),cmap = 'gray')
    #plt.show()

