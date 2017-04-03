from lane import Lane
from dotted_detect import detect_dotted
from detect_color import detect_color

class LaneTypeAnalysis:
    def __init__(self):
        pass

def analyze_lane_type(img, img_color, last_id_left, last_id_right):
	right_lane = Lane()
	left_lane = Lane()

	left_lane.solid, right_lane.solid = detect_dotted(img)
	if (left_lane.solid == -1 or right_lane.solid == -1):
		return last_id_left, last_id_right
	left_lane.color, right_lane.color = detect_color(img_color)

	return left_lane, right_lane