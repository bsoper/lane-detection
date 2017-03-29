from lane import Lane
from dotted_detect import detect_dotted
from detect_color import detect_color

def analyze_lane_type(img, img_color):
	right_lane = Lane()
	left_lane = Lane()

	left_lane.solid, right_lane.solid = detect_dotted(img)
	left_lane.color, right_lane.color = detect_color(img_color)

	return left_lane, right_lane