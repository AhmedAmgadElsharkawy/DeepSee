import cv2
import numpy as np
from scipy.ndimage import convolve
import skimage.exposure as exposure
class EdgeDetectionController():
    def __init__(self,edge_detection_window):
        self.edge_detection_window = edge_detection_window
        self.edge_detection_window.apply_button.clicked.connect(self.apply_edge_detection)

    def apply_edge_detection(self):
        type = self.edge_detection_window.edge_detector_type_custom_combo_box.current_text()
        image = self.edge_detection_window.input_image_viewer.image_model.get_image_matrix()
        if type == "Sobel Detector":
            result = self.sobel(image)
        elif type == "Roberts Detector":
            result = self.roberts(image)
        elif type == "Prewitt Detector":
            result = self.prewitt(image)
        else:
            result = self.canny(image)
        self.edge_detection_window.output_image_viewer.display_and_set_image_matrix(result)
    
    def prewitt(self, image):
        kernel_size = self.edge_detection_window.prewitt_detector_kernel_size_spin_box.value()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        if kernel_size == 3:
            prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        else:
            prewitt_kernel_x = np.array([[2, 1, 0, -1, -2],
                            [2, 1, 0, -1, -2],
                            [2, 1, 0, -1, -2],
                            [2, 1, 0, -1, -2],
                            [2, 1, 0, -1, -2]])

            prewitt_kernel_y = np.array([[2, 2, 2, 2, 2],
                            [1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0],
                            [-1, -1, -1, -1, -1],
                            [-2, -2, -2, -2, -2]])
        prewitt_x = self.convolution(image, prewitt_kernel_x)
        prewitt_y = self.convolution(image, prewitt_kernel_y)
        magnitude = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
        prewitt_magnitude = (
            exposure.rescale_intensity(magnitude, in_range="image", out_range=(0, 255))
            .clip(0, 255)
            .astype(np.uint8)
        )
        return prewitt_magnitude
    
    def canny(self, image):
        kernel_size = self.edge_detection_window.canny_detector_kernel_spin_box.value()
        lower_threshold = self.edge_detection_window.canny_detector_lower_threshold_spin_box.value()
        upper_threshold = self.edge_detection_window.canny_detector_upper_threshold_spin_box.value()
        variance = self.edge_detection_window.canny_detector_variance_spin_box.value()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), variance)

        edges = cv2.Canny(image, lower_threshold, upper_threshold)
        return edges

    def sobel(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = self.edge_detection_window.sobel_detector_kernel_size_spin_box.value()
        direction = self.edge_detection_window.sobel_detector_direction_custom_combo_box.current_text()
        sigma = 0
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        if kernel_size == 3:
            sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            sobel_kernel_x = np.array([[2, 1, 0, -1, -2],
                            [3, 2, 0, -2, -3],
                            [4, 3, 0, -3, -4],
                            [3, 2, 0, -2, -3],
                            [2, 1, 0, -1, -2]])
            sobel_kernel_y = np.array([[2, 3, 4, 3, 2],
                            [1, 2, 3, 2, 1],
                            [0, 0, 0, 0, 0],
                            [-1, -2, -3, -2, -1],
                            [-2, -3, -4, -3, -2]])
        sobel_x = self.convolution(image, sobel_kernel_x)
        sobel_y = self.convolution(image, sobel_kernel_y)

        phase = np.rad2deg(np.arctan2(sobel_y, sobel_x))
        phase[phase < 0] += 180

        if direction == "Horizontal":
            sobel_x_normalized = (
                exposure.rescale_intensity(
                    sobel_x, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_x_normalized
        elif direction == "Vertical":
            sobel_y_normalized = (
                exposure.rescale_intensity(
                    sobel_y, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_y_normalized
        elif direction == "Combined":
            sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            sobel_magnitude = (
                exposure.rescale_intensity(
                    sobel_magnitude, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            return sobel_magnitude
        else:
            raise ValueError("Invalid direction. Please use x, y or both.")
        
    def roberts(self, image):
        """Applies the Roberts Cross edge detection filter."""
        kernel_size = self.edge_detection_window.roberts_detector_kernel_size_spin_box.value()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Define Roberts kernels
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])

        # Apply convolution
        grad_x = self.convolution(image, roberts_x)
        grad_y = self.convolution(image, roberts_y)

        # Compute gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = (
                exposure.rescale_intensity(
                    magnitude, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

        return magnitude
        
    def convolution(self, image, kernel):
        img_height, img_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Compute padding size
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        # Pad the image (zero-padding)
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        # Initialize output image
        output = np.zeros((img_height, img_width), dtype=np.float32)

        # Perform convolution
        for i in range(img_height):
            for j in range(img_width):
                # Extract the region
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                # Compute the sum of element-wise multiplication
                output[i, j] = np.sum(region * kernel)

        return output