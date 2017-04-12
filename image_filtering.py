import cv2
import numpy as np


def verify_image_filter(img_filter):
    if img_filter is None:
        return img_filter

    width = img_filter.size
    half_width = width / 2

    left_side = img_filter[:half_width]
    right_side = img_filter[half_width:]

    if left_side.sum() / left_side.size < 0.05:
        img_filter[:half_width] = np.ones((half_width,))
    if right_side.sum() / right_side.size < 0.05:
        img_filter[half_width:] = np.ones((half_width,))

    return img_filter


def filter_image(img, old_mask=None):
    amount = 1.0 if img.max() == 1.0 else 255.0
    vertical_sum = np.sum(img, axis=0) / (amount * np.shape(img)[0])

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
