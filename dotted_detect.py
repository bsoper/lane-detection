import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane

def detect_dotted(img):
    img = cv2.imread(img,0)
    median = cv2.medianBlur(img,11)
    edges = cv2.Canny(median,10,20)
    # im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(median, contours, 0, (128,255,0), 3)
    cnts = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    height, width = img.shape
    counts = {'left': 0, 'right': 0}
    on_left = True

    for c in cnts:
      # compute the center of the contour
      M = cv2.moments(c)
      area = int(M['m00'])
      if area == 0:
        continue
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      # draw the contour and center of the shape on the image

      if area > 2000:
        if cX > width / 2:
          counts['right'] += 1
          on_left = True
        else:
          counts['left'] += 1
          on_left = False

    if counts['left'] > 1:
      left_solid = 'Dash'
    elif counts['left'] == 1:
      left_solid= 'Solid'

    if counts['right'] > 1:
      right_solid = 'Dash'
    elif counts['right'] == 1:
      right_solid = 'Solid'
    
    return left_solid, right_solid
         

if __name__ == '__main__':
    left, right = detect_dotted('lane_type_test/warped3.jpg')
    #plt.imshow(detect_dotted('warped3.jpg'),cmap = 'gray')
    #plt.show()

