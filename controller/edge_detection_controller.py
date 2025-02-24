import cv2
import numpy as np
class EdgeDetectionController():
    def __init__(self,edge_detection_window):
        self.edge_detection_window = edge_detection_window
        self.edge_detection_window.apply_button.clicked.connect(self.apply_edge_detection)

    def apply_edge_detection(self):
        type = self.edge_detection_window.edge_detector_type_custom_combo_box.current_text()
        if type == "Sobel Detector":
            result = self.sobel(self.edge_detection_window.input_image_viewer.image_model.get_image_matrix())
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