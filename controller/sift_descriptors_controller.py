import cv2
import numpy as np
from math import floor
from cv2 import KeyPoint


class SiftDescriptorsController():
    def __init__(self,sift_descriptors_window):
        self.sift_descriptors_window = sift_descriptors_window
        self.sift_descriptors_window.apply_button.clicked.connect(self.apply_sift)

    def apply_sift(self):
        image=self.sift_descriptors_window.input_image_viewer.image_model.get_gray_image_matrix()

        gaussian_pyramid=self.generate_gaussian_pyramid(image)
        dog_pyramid=self.generateDoGImages(gaussian_pyramid)
        keypoints=self.detect_keypoints(dog_pyramid) #keypoint information (octave, scale, and coordinates) to the keypoints list.
        refined_keypoints = self.localize_keypoints(keypoints, dog_pyramid)
        oriented_keypoints = self.assign_orientations(gaussian_pyramid, refined_keypoints)
        self.sift_descriptors_window.output_image_viewer.display_and_set_image_matrix(dog_pyramid[0][1])

        print("sift")
    def gaussian_blur(self,image,sigma):
        filters = self.sift_descriptors_window.main_window.filters_window.filters_controller
        image=filters.gaussian_filter(image, kernel_size=3, sigma=sigma)
        return image

    def generate_gaussian_pyramid(self,image, num_octaves=4, num_scales=5, sigma=1.6):
        """Generates a Gaussian pyramid with different levels of blur."""
        k = 2 ** (1.0 / (num_scales - 1))  # Scale multiplier
        pyramid = []

        for octave in range(num_octaves):
            octave_images = []
            for scale in range(num_scales):
                sigma_scaled = sigma * (k ** scale)
                print(image.shape)
                blurred = self.gaussian_blur(image, sigma_scaled)
                octave_images.append(blurred)
            pyramid.append(octave_images)
            image = cv2.pyrDown(image)  # Downsample for next octave

        return pyramid
    def compute_dog_pyramid(self,gaussian_pyramid):
        """Computes the Difference of Gaussians (DoG) by subtracting consecutive blurred images."""
        dog_pyramid = []

        for octave in gaussian_pyramid:
            dog_octave = []
            for i in range(1, len(octave)):
                dog_octave.append(cv2.subtract(octave[i], octave[i-1]))
            dog_pyramid.append(dog_octave)

        return dog_pyramid

    def generateDoGImages(self, gaussian_images):
        """Generate Difference-of-Gaussians image pyramid
        """
        dog_images = []

        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(np.subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
            dog_images.append(dog_images_in_octave)
        return dog_images

    def detect_keypoints(self,dog_pyramid):
        """Detects keypoints by finding local extrema in the DoG pyramid."""
        keypoints = []

        for o, dog_octave in enumerate(dog_pyramid):
            for i in range(1, len(dog_octave) - 1):  # Avoid first and last scale
                prev_img, curr_img, next_img = dog_octave[i-1], dog_octave[i], dog_octave[i+1]

                for y in range(1, curr_img.shape[0] - 1):
                    for x in range(1, curr_img.shape[1] - 1):
                        pixel = curr_img[y, x]

                        neighbors = np.concatenate([
                            prev_img[y-1:y+2, x-1:x+2].flatten(),
                            curr_img[y-1:y+2, x-1:x+2].flatten(),
                            next_img[y-1:y+2, x-1:x+2].flatten()
                        ])

                        if pixel == np.max(neighbors) or pixel == np.min(neighbors):
                            keypoints.append((o, i, x, y))

        return keypoints
    
    def localize_keypoints(self, keypoints, dog_pyramid, contrast_threshold=0.03):
        """Filters out low-contrast keypoints."""
        refined_keypoints = []

        for (o, i, x, y) in keypoints:
            pixel_value = dog_pyramid[o][i][y, x]
            if abs(pixel_value) > contrast_threshold:
                refined_keypoints.append((o, i, x, y))

        return refined_keypoints
    
    def assign_orientations(self, gaussian_pyramid, keypoints):
        """Assigns orientations to keypoints using gradient histograms."""
        oriented_keypoints = []

        for (o, i, x, y) in keypoints:
            image = gaussian_pyramid[o][i]
            if y < 1 or y >= image.shape[0] - 1 or x < 1 or x >= image.shape[1] - 1:
                continue

            # Compute gradients
            dx = image[y, x+1] - image[y, x-1]
            dy = image[y-1, x] - image[y+1, x]

            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = (np.arctan2(dy, dx) * 180 / np.pi) % 360

            oriented_keypoints.append((o, i, x, y, magnitude, orientation))

        return oriented_keypoints
