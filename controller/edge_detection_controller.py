import cv2
import numpy as np
from scipy.ndimage import convolve
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
            print("ROB")
        elif type == "Prewitt Detector":
            result = self.prewitt(image)
        self.edge_detection_window.output_image_viewer.display_image_matrix(result)

    def sobel(self, image):
        # Apply Sobel operator in X and Y directions
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X-direction
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y-direction

        # Compute the gradient magnitude
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)

        # Convert to 8-bit for visualization
        sobel_combined = np.uint8(sobel_combined)
        return sobel_combined
    
    def prewitt(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)  # Horizontal edges
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)  # Vertical edges

        # Apply convolution with Prewitt kernels
        edge_x = convolve(gray_image, prewitt_x)
        edge_y = convolve(gray_image, prewitt_y)

        # Compute gradient magnitude
        prewitt_combined = np.sqrt(edge_x**2 + edge_y**2)
        prewitt_combined = (prewitt_combined / np.max(prewitt_combined) * 255).astype(np.uint8) 
        return prewitt_combined