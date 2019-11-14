#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


def gray_scale(img):
    assert(img.ndim == 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def normalize(mean, std, data):
    return (data - mean) / std


def rotate(img, angle_range_val=10):
    h, w = img.shape[:2]
    angle_ranges = np.random.uniform(-angle_range_val, angle_range_val)
    rotation_matrix = cv2.getRotationMatrix2D(
        (w // 2, h // 2), angle_ranges, 1)

    return cv2.warpAffine(img, rotation_matrix, (w, h))


def random_crop(img, factor=5):
    assert factor > 0

    top_left = np.random.randint(0, factor, 2)
    bottom_right = np.random.randint(1, factor, 2)

    cropped_img = np.copy(img)
    cropped_img = cropped_img[
        top_left[0]: -bottom_right[0], top_left[1]: -bottom_right[1]
    ]

    return cv2.resize(cropped_img, img.shape[:2])


def cutout(img, mask_size=6, mask_value="mean", p=0.5):
    cutout_img = np.copy(img)

    h, w = img.shape[:2]

    if mask_value == "mean":
        mask_value = cutout_img.mean()
    elif mask_value == "random":
        mask_value = np.random.randint(0, 256)

    top_left = np.random.randint(-mask_size // 2, h - mask_size, 2)
    bottom_right = top_left + mask_size

    if top_left[0] < 0:
        top_left[0] = 0
    if top_left[1] < 0:
        top_left[1] = 0

    cutout_img[top_left[0]: bottom_right[0],
               top_left[1]: bottom_right[1], :].fill(mask_value)

    return cutout_img
