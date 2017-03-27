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
    right = Lane()
    left = Lane()

    for c in cnts:
      # compute the center of the contour
      M = cv2.moments(c)
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      area = int(M['m00'])
      # draw the contour and center of the shape on the image

      if area > 1000:
        if cX > width / 2:
          counts['right'] += 1
          on_left = True
        else:
          counts['left'] += 1
          on_left = False

        # cv2.drawContours(median, [c], -1, (180, 255, 10), 2)
        # cv2.circle(median, (cX, cY), 7, (255, 255, 255), -1)
        #cv2.putText(median, "DASH", (cX - 100, cY - 20),
        #  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
      #if area >= 5000:
        #cv2.putText(median, "SOLID", (cX - 100, cY - 20),
        #  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
      #print (int(height / 2))
    if counts['left'] > 1:
      cv2.putText(median, "DASH", (int(width / 6), int(height / 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
      left.solid = 'Dash'
    elif counts['left'] == 1:
      cv2.putText(median, "Solid", (int(width / 6), int(height / 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
      left.solid = 'Solid'

    if counts['right'] > 1:
      cv2.putText(median, "DASH", (int(5 * width / 6), int(height / 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
      right.solid = 'Dash'
    elif counts['right'] == 1:
      cv2.putText(median, "Solid", (int(5 * width / 6), int(height / 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
      right.solid = 'Solid'
    
    return left, right
         
#    plt.subplot(211),plt.imshow(img,cmap = 'gray')
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(212),plt.imshow(median,cmap = 'gray')
#    plt.title('Blured Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(213),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#    plt.show()

if __name__ == '__main__':
    left, right = detect_dotted('lane_type_test/warped3.jpg')
    #plt.imshow(detect_dotted('warped3.jpg'),cmap = 'gray')
    #plt.show()

