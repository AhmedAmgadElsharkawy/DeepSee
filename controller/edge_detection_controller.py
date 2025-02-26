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
            result = self.sobel_edge_detection(image)
        elif type == "Roberts Detector":
            print("ROB")
        elif type == "Prewitt Detector":
            result = self.prewitt(image)
        else:
            result = self.canny(image)
        self.edge_detection_window.output_image_viewer.display_and_set_image_matrix(result)
    
    def prewitt(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = 3
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_y)
        magnitude = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
        prewitt_magnitude = (
            exposure.rescale_intensity(magnitude, in_range="image", out_range=(0, 255))
            .clip(0, 255)
            .astype(np.uint8)
        )
        return prewitt_magnitude
    
    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 10, 20)
        return edges

    def sobel_edge_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = 3
        sigma = 0
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_x = cv2.filter2D(image, cv2.CV_32F, sobel_x_kernel)
        sobel_y = cv2.filter2D(image, cv2.CV_32F, sobel_y_kernel)

        phase = np.rad2deg(np.arctan2(sobel_y, sobel_x))
        phase[phase < 0] += 180
        direction = "y"

        if direction == "x":
            sobel_x_normalized = (
                exposure.rescale_intensity(
                    sobel_x, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_x_normalized
        elif direction == "y":
            sobel_y_normalized = (
                exposure.rescale_intensity(
                    sobel_y, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_y_normalized
        elif direction == "both":
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