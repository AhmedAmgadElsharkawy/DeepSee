import cv2
import numpy as np
class EdgeDetectionController():
    def __init__(self,edge_detection_window):
        self.edge_detection_window = edge_detection_window
        self.edge_detection_window.apply_button.clicked.connect(self.apply_edge_detection)

    def apply_edge_detection(self):
        print("applied")

    def sobel(self, image):
        # Apply Sobel operator in X and Y directions
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X-direction
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y-direction

        # Compute the gradient magnitude
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)

        # Convert to 8-bit for visualization
        sobel_combined = np.uint8(sobel_combined)