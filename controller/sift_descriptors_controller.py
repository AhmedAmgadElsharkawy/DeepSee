import cv2
import numpy as np
from math import floor
from cv2 import KeyPoint


class SiftDescriptorsController():
    def __init__(self,sift_descriptors_window):
        self.sift_descriptors_window = sift_descriptors_window
        self.sift_descriptors_window.apply_button.clicked.connect(self.apply_sift)

    def apply_sift(self):
        input_image = self.sift_descriptors_window.input_image_viewer.image_model.get_image_matrix()
        print(input_image.shape)
        image=self.sift_descriptors_window.input_image_viewer.image_model.get_gray_image_matrix()
        print(image.shape)

        gaussian_pyramid=self.generate_gaussian_pyramid(image)
        dog_pyramid=self.generateDoGImages(gaussian_pyramid)
        print(dog_pyramid)
        keypoints=self.findScaleSpaceExtrema(gaussian_pyramid, dog_pyramid) #keypoint information (octave, scale, and coordinates) to the keypoints list.
        print("KeyPoints: ", keypoints)
        self.sift_descriptors_window.detect_keypoints_output_image_viewer.display_and_set_image_matrix(dog_pyramid[0][1])

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

    def findScaleSpaceExtrema(self, gaussian_images, dog_images, num_intervals=1, sigma=1.6, image_border_width=1, contrast_threshold=0.04):
        """Find pixel positions of all scale-space extrema in the image pyramid
        """
        threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
        keypoints = []
        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
                # (i, j) is the center of the 3x3 array
                for i in range(1, second_image.shape[0] - 1):
                    for j in range(1, second_image.shape[1] - 1):
                        if self.isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                            localization_result = self.localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                                for keypoint_with_orientation in keypoints_with_orientations:
                                    keypoints.append(keypoint_with_orientation)
        return keypoints

    def isPixelAnExtremum(self, first_subimage, second_subimage, third_subimage, threshold):
        """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
        """
        center_pixel_value = second_subimage[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= first_subimage) and \
                    np.all(center_pixel_value >= third_subimage) and \
                    np.all(center_pixel_value >= second_subimage[0, :]) and \
                    np.all(center_pixel_value >= second_subimage[2, :]) and \
                    center_pixel_value >= second_subimage[1, 0] and \
                    center_pixel_value >= second_subimage[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= first_subimage) and \
                    np.all(center_pixel_value <= third_subimage) and \
                    np.all(center_pixel_value <= second_subimage[0, :]) and \
                    np.all(center_pixel_value <= second_subimage[2, :]) and \
                    center_pixel_value <= second_subimage[1, 0] and \
                    center_pixel_value <= second_subimage[1, 2]
        return False

    def localizeExtremumViaQuadraticFit(self, i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
        """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
        """
        extremum_is_outside_image = False
        image_shape = dog_images_in_octave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
            pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                                second_image[i-1:i+2, j-1:j+2],
                                third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.computeGradientAtCenterPixel(pixel_cube)
            hessian = self.computeHessianAtCenterPixel(pixel_cube)
            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the image
            if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
            hessian = np.array(hessian)
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV KeyPoint object
                keypoint = KeyPoint()
                keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(functionValueAtUpdatedExtremum)
                return keypoint, image_index
        return None

    def computeGradientAtCenterPixel(self, pixel_array):
        """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        """
        # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
        # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return [dx, dy, ds]

    def computeHessianAtCenterPixel(self, pixel_array):
        """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        """
        # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
        # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
        # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
        # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return [[dxx, dxy, dxs], 
                    [dxy, dyy, dys],
                    [dxs, dys, dss]]
    
    def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        """Compute orientations for each keypoint
        """
        logger.debug('Computing keypoint orientations...')
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = zeros(num_bins)
        smooth_histogram = zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx))
                        weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                        histogram_index = int(round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                # Quadratic peak interpolation
                # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < float_tolerance:
                    orientation = 0
                new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations