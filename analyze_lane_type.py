from lane import Lane
from dotted_detect import detect_dotted
from detect_color import detect_color
import numpy as np


class LaneTypeAnalysis:
    def __init__(self):
        self.new_count_left = 0
        self.new_count_right = 0
        self.last_left = Lane()
        self.last_right = Lane()
        self.first_frame = True
        self.left_mask = None
        self.right_mask = None
        self.left_fit = None
        self.right_fit = None

    def get_lane_type(self, img, color_img):
        right_lane = Lane()
        left_lane = Lane()

        left_lane.color, right_lane.color = detect_color(color_img)
        left_lane, right_lane, left_centers, right_centers, l_mask, r_mask = detect_dotted(img,
                                                                                           left_lane,
                                                                                           right_lane,
                                                                                           self.left_mask,
                                                                                           self.right_mask)

        self.left_mask = l_mask
        self.right_mask = r_mask
        #print (right_lane.single)

        # handle errors
        if left_lane != self.last_left:
            if self.new_count_left >= 5 or self.first_frame:
                self.last_left = left_lane
                self.new_count_left = 0
            else:
                left_lane = self.last_left
                self.new_count_left += 1
        else:
            self.new_count_left = 0
        if right_lane != self.last_right:
            if self.new_count_right >= 5 or self.first_frame:
                self.last_right = right_lane
                self.new_count_right = 0
                self.first_frame = False
            else:
                right_lane = self.last_right
                self.new_count_right += 1
        else:
            self.new_count_right = 0

        return left_lane, right_lane, left_centers, right_centers

    def update_polyfit_coeff(self, left_fit, right_fit):
        self.left_fit = left_fit
        self.right_fit = right_fit
