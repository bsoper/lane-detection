import cv2
from lane import Lane
from dotted_detect import detect_dotted
from detect_color import detect_color

def analyze_lane_type(img, img_color, last_id_left, last_id_right):
	right_lane = Lane()
	left_lane = Lane()

	img_read = cv2.imread(img,0)
	left_img = img_read[0:img_read.shape[0], 0:img_read.shape[1]/2]
	right_img = img_read[0:img_read.shape[0], img_read.shape[1]/2:img_read.shape[1]]

	left_lane.solid = detect_dotted(left_img)
	right_lane.solid = detect_dotted(right_img)

	if (left_lane.solid == -1 or right_lane.solid == -1):
		return last_id_left, last_id_right

	img = cv2.imread(img_color, cv2.IMREAD_COLOR)
	left_color_img = img[0:img.shape[0], 0:img.shape[1]/2]
	right_color_img = img[0:img.shape[0], img.shape[1]/2:img.shape[1]]

	left_lane.color = detect_color(left_color_img)
	right_lane.color = detect_color(right_color_img)

	return left_lane, right_lane