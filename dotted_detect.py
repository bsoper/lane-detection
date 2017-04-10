import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from lane import Lane
from scipy import misc
import sys
from random import randint


def detect_dotted(img, left_lane, right_lane, left_side_mask, right_side_mask):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    left_side = img[:, 0:width / 2]
    right_side = img[:, width / 2:]

    left_side, left_side_mask = filter_image(left_side, left_side_mask)
    right_side, right_side_mask = filter_image(right_side, right_side_mask)

    left_lane.solid, left_lane_info = detect_dotted_side(left_side)
    # left_lane_info = filter_lane_info(left_lane_info)
    left_width = calculate_lane_width(left_lane_info)
    right_lane.solid, right_lane_info = detect_dotted_side(right_side)
    #print (filter_lane_info(left_lane_info))
    # right_lane_info = filter_lane_info(right_lane_info)
    right_width = calculate_lane_width(right_lane_info)

    ratio = left_width / right_width
    left_lane.single = 'Single'
    right_lane.single = 'Single'
    if ratio > 1.5:
        left_lane.single  = 'Double'
    elif ratio < 0.666:
        right_lane.single  = 'Double'

    left_centers = [info[0] for info in left_lane_info]
    right_centers = [info[0] for info in right_lane_info]

    return left_lane, right_lane, left_centers, right_centers, left_side_mask, right_side_mask


def calculate_lane_width(lane_info):
    widths = [info[1] for info in lane_info]
    if len(widths) == 0:
        return 0
    else:
        return sum(widths) / len(widths)

def filter_lane_info(lane_info):
    center_initial = sum([info[0][0] for info in lane_info]) / len(lane_info)
    thresh = calculate_lane_width(lane_info)
    filtered = [info for info in lane_info if abs(center_initial - info[0][0]) < thresh]
    #print (center_initial, thresh)
    return filtered


def filter_image(img, old_mask=None):
    vertical_sum = np.sum(img, axis=0) / (255 * np.shape(img)[0])

    threshold_mask = np.asarray([int(val) for val in vertical_sum > 0.03])
    if old_mask is not None:
        kernel = np.ones((np.ceil(img.shape[1]/40), np.ceil(img.shape[1]/40)), np.uint8)
        old_mask = cv2.morphologyEx((old_mask * 1.0).astype(np.float32), cv2.MORPH_DILATE, kernel).reshape(threshold_mask.shape[0])
        new_mask = threshold_mask * old_mask
    else:
        new_mask = threshold_mask

    full_mask = np.tile(new_mask, (img.shape[0], 1))

    img = full_mask * img

    return img, new_mask


def detect_dotted_side(img):
    height, width = img.shape
    row_sum = np.sum(img, axis=1)
    top_img_nonzero = np.where(row_sum > 100)[0][0]
    lane_info = []
    num_zero = 0

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
    # plt.imshow(detect_dotted('warped3.jpg'),cmap = 'gray')
    # plt.show()
