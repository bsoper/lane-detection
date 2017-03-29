import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane

def detect_color(img):
	img = cv2.imread(img,cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	height, width, pix_dim = img.shape

	left_side = img[:, 0:width/2]
	right_side = img[:, width/2:width]

	#print (np.where(left_side != [0,0,0]))
	print (left_side[left_side != [0,0,0]])
	left_avg = np.mean(left_side[np.nonzero(left_side)], axis=(0,1))
	right_avg = np.mean(right_side[np.nonzero(right_side)], axis=(0,1))

	#print(left_side[np.nonzero(left_side)])

	print (left_avg, right_avg)

