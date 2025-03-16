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
                output_image_matrix = self.detect_circles(input_image_matrix, min_radius = min_radius, max_radius = max_radius, color = color)
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
        thetas = np.linspace(-np.pi/2, np.pi/2, num=theta)  # Range of theta values in radians

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
            rho_values = np.clip(np.round(rho_values).astype(int) + max_rho, 0, len(rhos) - 1)
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

    def detect_lines(self, input_image_matrix, rho=1, theta=np.pi / 360, threshold=90, color=(0, 0, 255)):
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
            gray = self.hough_transform_window.input_image_viewer.image_model.get_gray_image_matrix()
        else:
            gray = input_image_matrix.copy()  # Already grayscale, no conversion needed
        
        edges = cv2.Canny(gray, 50, 150)
        candidates_indices, rhos, thetas = self.find_lines(edges, rho, theta, threshold)
        result = self.draw_lines(input_image_matrix, candidates_indices, rhos, thetas, color)
        
        return result


    def draw_circles(self, input_image_matrix, circles, color):
        result = input_image_matrix.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                # Draw the outer circle
                cv2.circle(result, center, radius, color, 2)
                # Draw the center of the circle
                cv2.circle(result, center, 2, color, 3)

        return result
    
    def detect_circles(self, input_image_matrix, min_radius=10, max_radius=100, color=(0, 255, 0)):
        """
        Detect and draw circles in an image using the Hough Transform.

        Args:
            input_image_matrix (numpy.ndarray): Input image.
            min_radius (int): Minimum radius of circles to detect.
            max_radius (int): Maximum radius of circles to detect.
            color (tuple): Color of the circles to draw (BGR format).

        Returns:
            result (numpy.ndarray): Image with detected circles drawn.
        """
        if len(input_image_matrix.shape) == 3 and input_image_matrix.shape[2] == 3:
            gray = self.hough_transform_window.input_image_viewer.image_model.get_gray_image_matrix()
        else:
            gray = input_image_matrix.copy()  # Already grayscale, no conversion needed
        
        edges = cv2.Canny(gray, 50, 150)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        result = self.draw_circles(input_image_matrix, circles, color)

        return result

    def detect_ellipses(self,input_image_matrix):
        print("detect ellipses")
