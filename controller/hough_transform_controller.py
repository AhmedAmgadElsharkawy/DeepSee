import numpy as np
import cv2
from collections import defaultdict
from skimage.color import rgb2gray
import random
from skimage.feature import canny
import pandas as pd

class HoughTransformController():
    def __init__(self,hough_transform_window = None):
        self.hough_transform_window = hough_transform_window
        if self.hough_transform_window:
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
                output_image_matrix = self.detect_circles(input_image_matrix, min_radius = min_radius, max_radius = max_radius, threshold= threshold , color = color)
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
        thetas = np.linspace(-np.pi/2, np.pi/2, int(180 / theta))  # Range of theta values in radians

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
    
    
    def detect_circles(self, image, min_radius, max_radius, threshold, color = (0,0,255)):
        """
        Detects circles in an image using the Hough Transform algorithm.

        Args:
            image (numpy.ndarray): Input image (grayscale or BGR).
            min_radius (int): Minimum radius of circles to detect.
            max_radius (int): Maximum radius of circles to detect.
            threshold (float): Minimum vote percentage (relative to max votes) to consider a circle valid.
            color (tuple, optional): Color of the detected circles in BGR format (default is red (0,0,255)).

        Returns:
            numpy.ndarray: The input image with detected circles drawn on it.
            """
        post_process = True  # Flag to enable post-processing for filtering duplicate circles

        # Convert image to grayscale if it is colored (3-channel)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian Blur to reduce noise and improve edge detection
        gray = cv2.GaussianBlur(gray, (5, 5), 1)

        # Detect edges using Canny edge detector
        edges = cv2.Canny(gray, 50, 150)

        # Get edge points (nonzero pixels) as (y, x) coordinates
        edge_points = np.column_stack(np.where(edges > 0))

        # Get image dimensions
        img_height, img_width = edges.shape

        # Define theta range from 0 to 360 degrees (step of 10 degrees for efficiency)
        thetas = np.arange(0, 360, step=10)

        # Define radius range for circle detection
        rs = np.arange(min_radius, max_radius, step=1)

        # Precompute cosine and sine values for all theta values
        cos_thetas = np.cos(np.radians(thetas))
        sin_thetas = np.sin(np.radians(thetas))

        # Precompute circle candidate offsets for each radius value
        circle_candidates = [(r, (r * cos_thetas).astype(int), (r * sin_thetas).astype(int)) for r in rs]

        # Initialize a dictionary-based accumulator for voting
        accumulator = defaultdict(int)

        # Voting process: for each edge point, vote for possible circle centers
        for y, x in edge_points:
            for r, rcos_t, rsin_t in circle_candidates:
                # Compute candidate center coordinates
                x_center = x - rcos_t
                y_center = y - rsin_t

                # Ensure the candidate centers are within image boundaries
                valid_idx = (x_center >= 0) & (x_center < img_width) & (y_center >= 0) & (y_center < img_height)

                # Increment votes for valid circle centers
                for xc, yc in zip(x_center[valid_idx], y_center[valid_idx]):
                    accumulator[(xc, yc, r)] += 1

        # Copy the original image to draw detected circles
        output_img = image.copy()

        # Extract the circles with the highest votes
        out_circles = []
        max_votes = max(accumulator.values()) if accumulator else 1  # Avoid division by zero

        # Sort accumulator votes in descending order and apply thresholding
        for (x, y, r), votes in sorted(accumulator.items(), key=lambda i: -i[1]):
            current_vote_percentage = votes / max_votes
            if current_vote_percentage >= threshold:
                out_circles.append((x, y, r, current_vote_percentage))

        # Post-processing step to remove duplicate circles
        if post_process:
            pixel_threshold = 10  # Minimum distance between detected circles to avoid duplicates
            postprocessed_circles = []
            for x, y, r, v in out_circles:
                # Keep the circle if it's not too close to an already accepted one
                if all(np.linalg.norm(np.array([x, y]) - np.array([xc, yc])) > pixel_threshold for xc, yc, rc, v in postprocessed_circles):
                    postprocessed_circles.append((x, y, r, v))
            out_circles = postprocessed_circles  # Update the final list of circles


        for x, y, r, _ in out_circles:
            cv2.circle(output_img, (x, y), r, color, 2)

        return output_img 
    

    def detect_ellipses(self,input_image_matrix):
        original = input_image_matrix.copy()
        ellipse_canny_sigma = self.hough_transform_window.ellipses_detection_canny_sigma_spin_box.value()
        ellipse_canny_low_threshold = self.hough_transform_window.ellipses_detection_canny_low_threshold_spin_box.value()
        ellipse_canny_high_threshold = self.hough_transform_window.ellipses_detection_canny_high_threshold_spin_box.value()
        number_of_detected_ellipses = self.hough_transform_window.number_of_detected_ellipses_spin_box.value()

        if input_image_matrix.shape[-1] == 4:  
            input_image_matrix = input_image_matrix[:, :, :3] 


        if len(input_image_matrix.shape) == 3 and input_image_matrix.shape[2] == 3:  
            input_image_matrix = rgb2gray(input_image_matrix)  


        if input_image_matrix.dtype == np.float64:
            input_image_matrix = (input_image_matrix * 255).astype(np.uint8)

        accumulator = []
        
        edge = self.canny_edge_detector(input_image_matrix, sigma = ellipse_canny_sigma, low_threshold = ellipse_canny_low_threshold, high_threshold = ellipse_canny_high_threshold)
        pixels = np.array(np.where(edge == 255)).T
        edge_pixels = [p for p in pixels]

        if(len(edge_pixels) < 3):
            print("Not Enough Edge Pixels")
            return None

        # if len(edge_pixels):
        #     print(len(edge_pixels))
        #     edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for display
        #     return edge_bgr

        max_iter = max(5000,len(edge_pixels))

        for count, i in enumerate(range(max_iter)):
            p1, p2, p3 = self.randomly_pick_ellipse_point(edge_pixels)
            point_package = [p1, p2, p3]

            center = self.find_ellipse_center(point_package,edge)

            if center is None or self.valid_image_point(center,edge):
                continue

            major_axis, minor_axis, angle = self.find_ellipse_axes(point_package, center)

            if (major_axis is None) and (minor_axis is None):
                continue


            similar_idx = self.similar_ellipse(center[0], center[1], major_axis, minor_axis, angle,accumulator)
            if similar_idx == -1:
                score = 1
                accumulator.append([center[0], center[1], major_axis, minor_axis, angle, score])
            else:
                accumulator[similar_idx][-1] += 1
                weight = accumulator[similar_idx][-1]
                accumulator[similar_idx][0] = self.average_weight(accumulator[similar_idx][0], center[0], weight)
                accumulator[similar_idx][1] = self.average_weight(accumulator[similar_idx][1], center[1], weight)
                accumulator[similar_idx][2] = self.average_weight(accumulator[similar_idx][2], major_axis, weight)
                accumulator[similar_idx][3] = self.average_weight(accumulator[similar_idx][3], minor_axis, weight)
                accumulator[similar_idx][4] = self.average_weight(accumulator[similar_idx][4], angle, weight)

        accumulator = np.array(accumulator)
        df = pd.DataFrame(data=accumulator, columns=['x', 'y', 'axis1', 'axis2', 'angle', 'score'])
        accumulator = df.sort_values(by=['score'], ascending=False)
        

        accumulator = np.array(accumulator)
        df = pd.DataFrame(data=accumulator, columns=['x', 'y', 'axis1', 'axis2', 'angle', 'score'])
        accumulator = df.sort_values(by=['score'], ascending=False)

        top_ellipses = accumulator.iloc[:min(number_of_detected_ellipses,len(accumulator))].to_numpy()

        color = self.hex_to_bgr(self.hough_transform_window.choosen_color_hex)
        thickness = 2
        result_image = original
        if len(result_image.shape) == 2:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR) 

        # base_ellipse = top_ellipses[0]  
        # base_center = (int(base_ellipse[0]), int(base_ellipse[1]))
        # base_major = int(base_ellipse[2])
        # base_minor = int(base_ellipse[3])
        # base_angle = base_ellipse[4] * 180 / np.pi  

        for i, ellipse in enumerate(top_ellipses):
            p, q, a, b = map(int, np.around(ellipse[:4])) 
            angle = ellipse[4] * 180 / np.pi  
            score = ellipse[5] 
            cv2.ellipse(result_image, (p, q), (a, b), angle, 0, 360, color, thickness)

            # print(f"Ellipse {i}:")
            # print(f"  Center: ({p}, {q})")
            # print(f"  Major Axis: {a}")
            # print(f"  Minor Axis: {b}")
            # print(f"  Angle (degrees): {angle:.2f}")
            # print(f"  Score: {score:.2f}")  

            # if i > 0: 
            #     diff_center = np.sqrt((p - base_center[0])**2 + (q - base_center[1])**2)  
            #     diff_major = a - base_major
            #     diff_minor = b - base_minor
            #     diff_angle = angle - base_angle

                # print(f"  Center Difference: {diff_center:.2f}")
                # print(f"  Major Axis Difference: {diff_major}")
                # print(f"  Minor Axis Difference: {diff_minor}")
                # print(f"  Angle Difference (degrees): {diff_angle:.2f}")


        return result_image


    def canny_edge_detector(self, input_image_matrix,sigma,low_threshold,high_threshold):
        edged_image = canny(input_image_matrix, sigma = sigma, low_threshold = low_threshold, high_threshold = high_threshold)
        edge = np.zeros(edged_image.shape, dtype=np.uint8)
        edge[edged_image] = 255  
        return edge
    

    def randomly_pick_ellipse_point(self,edge_pixels):
        ran = random.sample(edge_pixels, 3)
        return (ran[0][1], ran[0][0]), (ran[1][1], ran[1][0]), (ran[2][1], ran[2][0])

    def valid_image_point(self, point,edge):
        if point[0] < 0 or point[0] >= edge.shape[1] or point[1] < 0 or point[1] >= edge.shape[0]:
            return True
        else:
            return False
        

    def find_ellipse_center(self, pt,edge):
        size = 7
        m, c = 0, 0
        tangents_slopes_array = []
        tangents_intercepts_array = []

        for i in range(len(pt)):
            xstart = pt[i][0] - size//2
            xend = pt[i][0] + size//2 + 1
            ystart = pt[i][1] - size//2
            yend = pt[i][1] + size//2 + 1
            crop = edge[ystart: yend, xstart: xend].T
            proximal_point = np.array(np.where(crop == 255)).T
            proximal_point[:, 0] += xstart
            proximal_point[:, 1] += ystart

            A = np.vstack([proximal_point[:, 0], np.ones(len(proximal_point[:, 0]))]).T
            m, c = np.linalg.lstsq(A, proximal_point[:, 1], rcond=None)[0]
            tangents_slopes_array.append(m)
            tangents_intercepts_array.append(c)
        slopes = []
        intercepts = []
        for i, j in zip([0, 1], [1, 2]):
            coefficients_matrix = np.array([[tangents_slopes_array[i], -1], [tangents_slopes_array[j], -1]])
            intercepts_vector = np.array([-tangents_intercepts_array[i], -tangents_intercepts_array[j]])
            det = np.linalg.det(coefficients_matrix)
            if abs(det) < 1e-10: 
                return None  
            tangents_intersection = np.linalg.solve(coefficients_matrix, intercepts_vector)
            middle_point = ((pt[i][0] + pt[j][0])/2, (pt[i][1] + pt[j][1])/2)
            denominator = middle_point[0] - tangents_intersection[0]

            if abs(denominator) < 1e-10:
                return None  
            slope = (middle_point[1] - tangents_intersection[1]) / denominator
            intercept = (middle_point[0]*tangents_intersection[1] - tangents_intersection[0]*middle_point[1]) / denominator
            slopes.append(slope)
            intercepts.append(intercept)

        coefficients_matrix = np.array([[slopes[0], -1], [slopes[1], -1]])
        intercepts_vector = np.array([-intercepts[0], -intercepts[1]])
        det = np.linalg.det(coefficients_matrix)
        if abs(det) < 1e-10: 
            return None  
        center = np.linalg.solve(coefficients_matrix, intercepts_vector)
        return center


    def find_ellipse_axes(self, points_package, center):
        offsets = []
        for point in points_package:
            offsets.append((point[0] - center[0], point[1] - center[1]))
        x1 = offsets[0][0]
        y1 = offsets[0][1]
        x2 = offsets[1][0]
        y2 = offsets[1][1]
        x3 = offsets[2][0]
        y3 = offsets[2][1]
        # general eq: A(x-center_x)^2 + B(x-center_x)(y-center_y)+C(y-center_y)^2 = 1
        coefficients_matrix = np.array([[x1**2, 2*x1*y1, y1**2], [x2**2, 2*x2*y2, y2**2], [x3**2, 2*x3*y3, y3**2]])
        constants_vector = np.array([1,1,1])
        det = np.linalg.det(coefficients_matrix)

        if abs(det) < 1e-10:  
            return None, None, None

        A, B, C = np.linalg.solve(coefficients_matrix, constants_vector)

        if self.valid_ellipse(A, B, C):
            angle = self.calculate_ellipse_rotation_angle(A, B, C)

            # using axes transformation and align the ellipse with the x-axis -> X = 1 / a²,  Y = 1 / b², X sin²θ + Y cos²θ = A, X cos²θ + Y sin²θ = C

            coefficients_matrix = np.array([[np.sin(angle) ** 2, np.cos(angle) ** 2], [np.cos(angle) ** 2, np.sin(angle) ** 2]])
            constants_matrix = np.array([A, C])
            det = np.linalg.det(coefficients_matrix)

            if abs(det) < 1e-10:  
                return None, None, None
            X , Y = np.linalg.solve(coefficients_matrix, constants_matrix)
            min_val = min(X, Y)

            if min_val <= 0:  
                return None, None, None
            major_axis = 1/np.sqrt(min(X,Y))
            minor_axis = 1/np.sqrt(max(X,Y))

            return major_axis, minor_axis, angle
        else:
            return None, None, None
        
    def valid_ellipse(self, a, b, c):
        if a*c - b**2 > 0:
            return True
        else:
            return False
        
    def calculate_ellipse_rotation_angle(self, a, b, c):
        angle = 0.5 * np.arctan2(2 * b, a - c) - np.pi / 2
        # Normalize the angle to the range [-90, 90]
        angle = np.rad2deg(angle) 
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        return np.deg2rad(angle)
    

    def similar_ellipse(self, p, q, axis1, axis2, angle,accumulator):
        similar_index = -1
        if accumulator is not None:
            for index, ellipse in enumerate(accumulator):
                center_difference = np.sqrt((ellipse[0] - p)**2 + (ellipse[1] - q)**2)
                angle_difference = (abs(ellipse[4] - angle))
                major_axis_difference = abs(max(axis1,axis2)-max(ellipse[2],ellipse[3]))
                minor_axis_difference = abs(min(axis1,axis2)-min(ellipse[2],ellipse[3]))
                if (major_axis_difference < 10) and (center_difference < 10) and ( angle_difference < 0.1745) and(minor_axis_difference < 10):
                    return index
        return similar_index
    

    def average_weight(self, old_value, new_value, score):
        return (old_value * score + new_value) / (score+1)