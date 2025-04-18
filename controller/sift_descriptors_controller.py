import cv2
import numpy as np
from math import floor
from cv2 import KeyPoint
from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging


class SiftDescriptorsController():
    def __init__(self,sift_descriptors_window):
        self.sift_descriptors_window = sift_descriptors_window
        self.sift_descriptors_window.apply_button.clicked.connect(self.apply_sift)

    def apply_sift(self):
        image = self.sift_descriptors_window.input_image_viewer.image_model.get_image_matrix()

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sigma, num_intervals, assumed_blur, image_border_width=1.6, 3, 0.5, 5
        gray_image = gray_image.astype('float32')
        
        base_image = self.generateBaseImage(gray_image, sigma, assumed_blur)

    def generateBaseImage(self, image, sigma, assumed_blur):
        """Generate base image from input image by upsampling by 2 in both directions and blurring
        """
        image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return self.gaussian_blur(image, sigma)  # the image blur is now sigma instead of assumed_blur

    def gaussian_blur(self,image,sigma):
        filters = self.sift_descriptors_window.main_window.filters_window.filters_controller
        image=filters.gaussian_filter(image, kernel_size=3, sigma=sigma)
        return image
