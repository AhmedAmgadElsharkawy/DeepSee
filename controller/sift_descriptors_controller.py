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
        num_octaves = self.computeNumberOfOctaves(base_image.shape)
        gaussian_kernels = self.generateGaussianKernels(sigma, num_intervals)
        gaussian_images = self.generateGaussianImages(base_image, num_octaves, gaussian_kernels)
        dog_images = self.generateDoGImages(gaussian_images)

    def generate_base_image(self, image, sigma, assumed_blur):
        """Generate base image from input image by upsampling by 2 in both directions and blurring
        """
        image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return self.gaussian_blur(image, sigma)  # the image blur is now sigma instead of assumed_blur

    def gaussian_blur(self,image,sigma):
        filters = self.sift_descriptors_window.main_window.filters_window.filters_controller
        image=filters.gaussian_filter(image, kernel_size=3, sigma=sigma)
        return image
    
    def compute_number_of_octaves(self, image_shape):
        """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
        """
        return int(round(log(min(image_shape)) / log(2) - 1))
    
    def generate_gaussian_kernels(self, sigma, num_intervals):
        """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
        """
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
        gaussian_kernels[0] = sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels

    def generate_gaussian_images(self, image, num_octaves, gaussian_kernels):
        """Generate scale-space pyramid of Gaussian images
        """
        gaussian_images = []

        for octave_index in range(num_octaves):
            gaussian_images_in_octave = []
            gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
            for gaussian_kernel in gaussian_kernels[1:]:
                image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-3]
            image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
        return gaussian_images
    
    def generate_DoG_images(self, gaussian_images):
        """Generate Difference-of-Gaussians image pyramid
        """
        dog_images = []

        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
            dog_images.append(dog_images_in_octave)
        return dog_images
