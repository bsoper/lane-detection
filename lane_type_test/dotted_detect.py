import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
     
img = cv2.imread('warped3.jpg',0)
median = cv2.medianBlur(img,11)
edges = cv2.Canny(median,10,20)
# im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(median, contours, 0, (128,255,0), 3)
cnts = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
  # compute the center of the contour
  M = cv2.moments(c)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  area = int(M['m00'])
  # draw the contour and center of the shape on the image

  if 5000 > area > 1000:
    # cv2.drawContours(median, [c], -1, (180, 255, 10), 2)
    # cv2.circle(median, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(median, "DASH", (cX - 100, cY - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
  if area >= 5000:
    cv2.putText(median, "SOLID", (cX - 100, cY - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

     
plt.subplot(211),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(median,cmap = 'gray')
plt.title('Blured Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(213),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()