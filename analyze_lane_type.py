from lane import Lane
from dotted_detect import detect_dotted
from detect_color import detect_color


class LaneTypeAnalysis:
    def __init__(self):
        self.new_count_left = 0
        self.new_count_right = 0
        self.last_left = Lane()
        self.last_right = Lane()
        self.first_frame = True

    def get_lane_type(self, img, color_img):
        right_lane = Lane()
        left_lane = Lane()

        left_lane.color, right_lane.color = detect_color(color_img)
        left_lane, right_lane, left_centers, right_centers = detect_dotted(img, left_lane, right_lane)
        #print (right_lane.single)

        # handle errors
        if left_lane != self.last_left:
            if self.new_count_left >= 30 or self.first_frame:
                self.last_left = left_lane
                self.new_count_left = 0
            else:
                left_lane = self.last_left
                self.new_count_left += 1
        else:
            self.new_count_left = 0
        if right_lane != self.last_right:
            if self.new_count_right >= 30 or self.first_frame:
                self.last_right = right_lane
                self.new_count_right = 0
                self.first_frame = False
            else:
                right_lane = self.last_right
                self.new_count_right += 1
        else:
            self.new_count_right = 0

        return left_lane, right_lane, left_centers, right_centers
