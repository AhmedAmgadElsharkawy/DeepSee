import cv2
import numpy as np
from scipy.ndimage import convolve
import skimage.exposure as exposure
import utils.utils as utils
class EdgeDetectionController():
    def __init__(self,edge_detection_window):
        self.edge_detection_window = edge_detection_window
        self.edge_detection_window.apply_button.clicked.connect(self.apply_edge_detection)

    def apply_edge_detection(self):
        type = self.edge_detection_window.edge_detector_type_custom_combo_box.current_text()
        # image = self.edge_detection_window.input_image_viewer.image_model.get_image_matrix()
        gray_image = self.edge_detection_window.input_image_viewer.image_model.get_gray_image_matrix()
        if type == "Sobel Detector":
            result, _ = self.canny_handmaded(gray_image)
        elif type == "Roberts Detector":
            result = self.roberts(gray_image)
        elif type == "Prewitt Detector":
            result = self.prewitt(gray_image)
        else:
            result = self.canny(gray_image)
        self.edge_detection_window.output_image_viewer.display_and_set_image_matrix(result)
    
    def prewitt(self, image):
        kernel_size = self.edge_detection_window.prewitt_detector_kernel_size_spin_box.value()
        image = self.edge_detection_window.main_window.filters_window.filters_controller.gaussian_filter(image, kernel_size)
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
        prewitt_x = utils.convolution(image, prewitt_kernel_x)
        prewitt_y = utils.convolution(image, prewitt_kernel_y)
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
        image = self.edge_detection_window.main_window.filters_window.filters_controller.gaussian_filter(image, kernel_size, variance)

        edges = cv2.Canny(image, lower_threshold, upper_threshold)
        return edges
    
    def canny_handmaded(self, image):
        kernel_size = self.edge_detection_window.canny_detector_kernel_spin_box.value()
        lower_threshold = self.edge_detection_window.canny_detector_lower_threshold_spin_box.value()
        upper_threshold = self.edge_detection_window.canny_detector_upper_threshold_spin_box.value()
        variance = self.edge_detection_window.canny_detector_variance_spin_box.value()

        magnitude, angles = self.sobel(image)
        image = self.non_maximum_suppression(magnitude, angles)



    def sobel(self, image):
        kernel_size = self.edge_detection_window.sobel_detector_kernel_size_spin_box.value()
        direction = self.edge_detection_window.sobel_detector_direction_custom_combo_box.current_text()
        image = self.edge_detection_window.main_window.filters_window.filters_controller.gaussian_filter(image, kernel_size)
        if kernel_size == 3:
            sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
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
        sobel_x = utils.convolution(image, sobel_kernel_x)
        sobel_y = utils.convolution(image, sobel_kernel_y)

        phase = np.rad2deg(np.arctan2(sobel_y, sobel_x))
        phase[phase < 0] += 180

        if direction == "Horizontal":
            sobel_x_normalized = self.normalize(sobel_x)
            return sobel_x_normalized, phase
        elif direction == "Vertical":
            sobel_y_normalized = self.normalize(sobel_y)
            return sobel_y_normalized, phase
        elif direction == "Combined":
            sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            sobel_magnitude = self.normalize(sobel_magnitude)
            return sobel_magnitude, phase
        
    def roberts(self, image):
        kernel_size = self.edge_detection_window.roberts_detector_kernel_size_spin_box.value()
        image = self.edge_detection_window.main_window.filters_window.filters_controller.gaussian_filter(image, kernel_size)

        roberts_x = np.array([[-1, 0], [0, 1]])
        roberts_y = np.array([[0, -1], [1, 0]])

        grad_x = utils.convolution(image, roberts_x)
        grad_y = utils.convolution(image, roberts_y)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = (
                exposure.rescale_intensity(
                    magnitude, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

        return magnitude
    
    def normalize(self, image):
        normalized = (
                exposure.rescale_intensity(
                    image, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

        return normalized
    
    def non_maximum_suppression(self, magnitude, angles):
        H, W = magnitude.shape
        suppressed = np.zeros((H, W), dtype=np.float32)

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                q = 255
                r = 255
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angles[i, j] < 67.5:
                    q = magnitude[i - 1, j + 1]
                    r = magnitude[i + 1, j - 1]
                elif 67.5 <= angles[i, j] < 112.5:
                    q = magnitude[i - 1, j]
                    r = magnitude[i + 1, j]
                elif 112.5 <= angles[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed
        
    