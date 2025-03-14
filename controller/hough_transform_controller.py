import numpy as np
import cv2
class HoughTransformController():
    def __init__(self,hough_transform_window):
        self.hough_transform_window = hough_transform_window
        self.hough_transform_window.apply_button.clicked.connect(self.apply_hough_transform)
    
    def apply_hough_transform(self):
        detected_objects_type = self.hough_transform_window.detected_objects_type_custom_combo_box.current_text()
        input_image_matrix = self.hough_transform_window.input_image_viewer.image_model.get_image_matrix()
        output_image_matrix = None
        theta = self.hough_transform_window.linear_detection_theta_custom_spin_box.value()
        threshold = self.hough_transform_window.linear_detection_threshold_custom_spin_box.value()
        color = self.hex_to_bgr(self.hough_transform_window.choosen_color_hex)
        min_radius = self.hough_transform_window.circles_detection_min_radius_spin_box.value()
        max_radius = self.hough_transform_window.circles_detection_max_radius_spin_box.value()
        
        if theta == 0:
            theta = 1
        if threshold == 0:
            threshold = 1

        match detected_objects_type:
            case "Lines Detection":
                output_image_matrix = self.detect_lines(input_image_matrix, theta = theta, threshold = threshold, color = color)
            case "Circles Detection":
                output_image_matrix = self.detect_circles(input_image_matrix)
            case "Ellipses Detection":
                output_image_matrix = self.detect_ellipses(input_image_matrix)
            

        if output_image_matrix is not None:
            self.hough_transform_window.output_image_viewer.display_and_set_image_matrix(output_image_matrix)

    def hex_to_bgr(self, hex_color):
            """
            Convert a hex color string to a BGR tuple.

            Args:
                hex_color (str): Hex color string (e.g., "#FF0000").

            Returns:
                tuple: BGR color tuple. Defaults to red (0, 0, 255) if input is invalid.
            """
            # Check if the input is a valid hex color string
            if not hex_color or not isinstance(hex_color, str) or len(hex_color) != 7 or not hex_color.startswith("#"):
                return (0, 0, 255)  # Default to red if input is invalid

            try:
                # Convert hex to BGR
                return tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))
            except ValueError:
                return (0, 0, 255)  # Default to red if conversion fails
    
    def find_lines(self, edges, rho=1, theta=np.pi / 180, threshold=100):
        """
        Detect lines in an edge-detected image using the Hough Transform.

        Args:
            edges (numpy.ndarray): Edge-detected image (binary image).
            rho (float): Distance resolution of the accumulator in pixels.
            theta (float): Angle resolution of the accumulator in radians.
            threshold (int): Accumulator threshold to detect a line.

        Returns:
            candidates_indices (numpy.ndarray): Indices of detected lines in the Hough accumulator.
            rhos (numpy.ndarray): Array of rho values.
            thetas (numpy.ndarray): Array of theta values.
        """
        height, width = edges.shape
        max_rho = int(np.sqrt(height**2 + width**2))  # Maximum possible rho value
        rhos = np.arange(-max_rho, max_rho + 1, rho)  # Range of rho values
        thetas = np.arange(0, np.pi, theta)  # Range of theta values in radians
        num_thetas = len(thetas)

        # Precompute cosine and sine values for all thetas
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        # Initialize the Hough accumulator
        accumulator = np.zeros((2 * len(rhos), num_thetas), dtype=np.uint64)

        # Get indices of edge pixels
        y_indices, x_indices = np.nonzero(edges)

        # Fill the accumulator
        for i in range(num_thetas):
            rho_values = x_indices * cos_thetas[i] + y_indices * sin_thetas[i]
            rho_values = np.round(rho_values).astype(int) + max_rho
            np.add.at(accumulator[:, i], rho_values, 1)

        # Find candidate lines that exceed the threshold
        candidates_indices = np.argwhere(accumulator >= threshold)
        candidate_values = accumulator[candidates_indices[:, 0], candidates_indices[:, 1]]
        sorted_indices = np.argsort(candidate_values)[::-1][: len(candidate_values)]
        candidates_indices = candidates_indices[sorted_indices]

        return candidates_indices, rhos, thetas

    def draw_lines(self, image, candidates_indices, rhos, thetas, color):
        """
        Draw detected lines on the input image.

        Args:
            image (numpy.ndarray): Input image.
            candidates_indices (numpy.ndarray): Indices of detected lines.
            rhos (numpy.ndarray): Array of rho values.
            thetas (numpy.ndarray): Array of theta values.
            color (tuple): Color of the lines to draw (BGR format).

        Returns:
            image (numpy.ndarray): Image with lines drawn.
        """
        for rho_idx, theta_idx in candidates_indices:
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
        return image

    def detect_lines(self, input_image_matrix, rho=1, theta=np.pi / 180, threshold=90, color=(0, 0, 255)):
        """
        Detect and draw lines in an image using the Hough Transform.

        Args:
            input_image_matrix (numpy.ndarray): Input image matrix.
            rho (float): Distance resolution of the accumulator in pixels.
            theta (float): Angle resolution of the accumulator in radians.
            threshold (int): Accumulator threshold to detect a line.
            color (tuple): Color of the lines to draw (BGR format).

        Returns:
            result (numpy.ndarray): Image with detected lines drawn.
        """
        if len(input_image_matrix.shape) == 3 and input_image_matrix.shape[2] == 3:
            gray = cv2.cvtColor(input_image_matrix, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_image_matrix.copy()  # Already grayscale, no conversion needed
        
        edges = cv2.Canny(gray, 50, 150)
        candidates_indices, rhos, thetas = self.find_lines(edges, rho, theta, threshold)
        result = self.draw_lines(input_image_matrix, candidates_indices, rhos, thetas, color)
        
        print(f"Detected {len(candidates_indices)} lines")
        for i, (rho_idx, theta_idx) in enumerate(candidates_indices[:10]):
            print(f"Line {i}: rho={rhos[rho_idx]}, theta={thetas[theta_idx]}")

        return result



    def detect_circles(self,input_image_matrix):
        print("detect circles")

    def detect_ellipses(self,input_image_matrix):
        print("detect ellipses")
