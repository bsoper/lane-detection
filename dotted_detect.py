import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane
from scipy import misc
import sys
from random import randint


def detect_dotted(img, left_lane, right_lane):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    left_side = img[:, 0:width / 2]
    right_side = img[:, width / 2:]

    left_lane.solid, left_lane_info = detect_dotted_side(left_side)
    left_width = calculate_lane_width(left_lane_info)
    right_lane.solid, right_lane_info = detect_dotted_side(right_side)
    right_width = calculate_lane_width(right_lane_info)

    # Update single vs double based on lane's relative width.
    ratio = left_width / right_width
    left_lane.single = 'Single'
    right_lane.single = 'Single'
    if ratio > 1.5:
        left_lane.single  = 'Double'
    elif ratio < 0.666:
        right_lane.single  = 'Double'

    left_centers = [info[0] for info in left_lane_info]
    right_centers = [info[0] for info in right_lane_info]

    return left_lane, right_lane, left_centers, right_centers

def calculate_lane_width(lane_info):
    widths = [info[1] for info in lane_info]
    if len(widths) == 0:
        return 0
    else:
        return sum(widths) / len(widths)

def detect_dotted_side(img):
    height, width = img.shape
    row_sum = np.sum(img, axis=1)
    top_img_nonzero = np.where(row_sum > 100)[0][0]
    lane_info = []
    num_zero = 0

    # Take cross sections. Save lane width and center information and
    # determine if it intersected a lane.
    for i in range(100):
        index = randint(top_img_nonzero,height-1)
        trimmed = np.trim_zeros(img[index,:])
        filter_indices = np.where(trimmed > 50)[0]
        trimmed = trimmed[filter_indices]
        lane_width = trimmed.shape[0]
        if lane_width < 5:
            num_zero += 1
        else:
            center_x = (filter_indices[0] + filter_indices[-1]) / 2
            lane_info.append(((center_x, index), lane_width))
    
    return ("Dash", "Solid")[int(num_zero < 25)], lane_info

if __name__ == '__main__':
    left = detect_dotted(sys.argv[1])
