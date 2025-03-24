import numpy as np
import cv2
import os
from itertools import product
from collections import defaultdict
from multiprocessing import Pool

from skimage.color import rgb2gray


import random
from skimage.feature import canny
import pandas as pd
import time

class HoughTransformController():
    def __init__(self,hough_transform_window):
        self.hough_transform_window = hough_transform_window
        self.hough_transform_window.apply_button.clicked.connect(self.apply_hough_transform)

        # self.major_bound = [100, 250]
        # self.minor_bound = [80, 250]
        # self.flattening_bound = 0.8

        # self.score_threshold = 7


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

        if input_image_matrix.shape[-1] == 4:  
            input_image_matrix = input_image_matrix[:, :, :3] 


        if len(input_image_matrix.shape) == 3 and input_image_matrix.shape[2] == 3:  
            input_image_matrix = rgb2gray(input_image_matrix)  


        if input_image_matrix.dtype == np.float64:
            input_image_matrix = (input_image_matrix * 255).astype(np.uint8)

        random.seed((time.time()*100) % 50)
        accumulator = []
        
        edge = self.canny_edge_detector(input_image_matrix, sigma = ellipse_canny_sigma, low_threshold = ellipse_canny_low_threshold, high_threshold = ellipse_canny_high_threshold)
        pixels = np.array(np.where(edge == 255)).T
        edge_pixels = [p for p in pixels]

        # if len(edge_pixels):
        #     print("Not Enough edge pixels")
        #     print(len(edge_pixels))
        #     edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for display
        #     return edge_bgr
        print("len(edge_pixels) = ",len(edge_pixels))

        max_iter = 5000

        for count, i in enumerate(range(max_iter)):
            p1, p2, p3 = self.randomly_pick_point(edge_pixels)
            point_package = [p1, p2, p3]

            center = self.find_center(point_package,edge)

            if center is None or self.point_out_of_image(center,edge):
                continue

            semi_axisx, semi_axisy, angle = self.find_semi_axis(point_package, center)
  

            if (semi_axisx is None) and (semi_axisy is None):
                continue

            # if not self.assert_diameter(semi_axisx, semi_axisy):
            #     continue

            similar_idx = self.is_similar(center[0], center[1], semi_axisx, semi_axisy, angle,accumulator)
            if similar_idx == -1:
                score = 1
                accumulator.append([center[0], center[1], semi_axisx, semi_axisy, angle, score])
            else:
                accumulator[similar_idx][-1] += 1
                w = accumulator[similar_idx][-1]
                accumulator[similar_idx][0] = self.average_weight(accumulator[similar_idx][0], center[0], w)
                accumulator[similar_idx][1] = self.average_weight(accumulator[similar_idx][1], center[1], w)
                accumulator[similar_idx][2] = self.average_weight(accumulator[similar_idx][2], semi_axisx, w)
                accumulator[similar_idx][3] = self.average_weight(accumulator[similar_idx][3], semi_axisy, w)
                accumulator[similar_idx][4] = self.average_weight(accumulator[similar_idx][4], angle, w)

        accumulator = np.array(accumulator)
        df = pd.DataFrame(data=accumulator, columns=['x', 'y', 'axis1', 'axis2', 'angle', 'score'])
        accumulator = df.sort_values(by=['score'], ascending=False)
        

        best = np.squeeze(np.array(accumulator.iloc[0]))
        p, q, a, b,angle = list(map(int, np.around(best[:5])))

        print("score: ", best[-1])
        print(p, q, a, b,angle)

        color = self.hex_to_bgr(self.hough_transform_window.choosen_color_hex)
        thickness = 2
        result_image = original
        if len(result_image.shape) == 2:
            result_image = cv2.cvtColor(input_image_matrix, cv2.COLOR_GRAY2BGR) 
        cv2.ellipse(result_image, (p, q), (a, b), angle * 180 / np.pi, 0, 360, color, thickness)

        return result_image  


    def canny_edge_detector(self, input_image_matrix,sigma,low_threshold,high_threshold):
        edged_image = canny(input_image_matrix, sigma = sigma, low_threshold = low_threshold, high_threshold = high_threshold)
        edge = np.zeros(edged_image.shape, dtype=np.uint8)
        edge[edged_image] = 255  
        return edge
    

    def randomly_pick_point(self,edge_pixels):
        ran = random.sample(edge_pixels, 3)
        return (ran[0][1], ran[0][0]), (ran[1][1], ran[1][0]), (ran[2][1], ran[2][0])

    def point_out_of_image(self, point,edge):
        if point[0] < 0 or point[0] >= edge.shape[1] or point[1] < 0 or point[1] >= edge.shape[0]:
            return True
        else:
            return False
        

    def find_center(self, pt,edge):
        size = 7
        m, c = 0, 0
        m_arr = []
        c_arr = []

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
            m_arr.append(m)
            c_arr.append(c)

        slope_arr = []
        intercept_arr = []
        for i, j in zip([0, 1], [1, 2]):
            coef_matrix = np.array([[m_arr[i], -1], [m_arr[j], -1]])
            dependent_variable = np.array([-c_arr[i], -c_arr[j]])
            det = np.linalg.det(coef_matrix)
            if abs(det) < 1e-10: 
                return None  
            t12 = np.linalg.solve(coef_matrix, dependent_variable)
            m1 = ((pt[i][0] + pt[j][0])/2, (pt[i][1] + pt[j][1])/2)
            slope = (m1[1] - t12[1]) / (m1[0] - t12[0])
            intercept = (m1[0]*t12[1] - t12[0]*m1[1]) / (m1[0] - t12[0])
            slope_arr.append(slope)
            intercept_arr.append(intercept)

        coef_matrix = np.array([[slope_arr[0], -1], [slope_arr[1], -1]])
        dependent_variable = np.array([-intercept_arr[0], -intercept_arr[1]])
        det = np.linalg.det(coef_matrix)
        if abs(det) < 1e-10: 
            return None  
        center = np.linalg.solve(coef_matrix, dependent_variable)
        return center


    def find_semi_axis(self, pt, center):
        npt = []
        for p in pt:
            npt.append((p[0] - center[0], p[1] - center[1]))
        x1 = npt[0][0]
        y1 = npt[0][1]
        x2 = npt[1][0]
        y2 = npt[1][1]
        x3 = npt[2][0]
        y3 = npt[2][1]
        coef_matrix = np.array([[x1**2, 2*x1*y1, y1**2], [x2**2, 2*x2*y2, y2**2], [x3**2, 2*x3*y3, y3**2]])
        dependent_variable = np.array([1,1,1])
        det = np.linalg.det(coef_matrix)

        if abs(det) < 1e-10:  
            return None, None, None

        A, B, C = np.linalg.solve(coef_matrix, dependent_variable)

        if self.assert_valid_ellipse(A, B, C):
            angle = self.calculate_ellipse_rotation_angle(A, B, C)

            AXIS_MAT = np.array([[np.sin(angle) ** 2, np.cos(angle) ** 2], [np.cos(angle) ** 2, np.sin(angle) ** 2]])
            AXIS_MAT_ANS = np.array([A, C])
            det = np.linalg.det(AXIS_MAT)

            if abs(det) < 1e-10:  
                return None, None, None
            X , Y = np.linalg.solve(AXIS_MAT, AXIS_MAT_ANS)
            major = 1/np.sqrt(min(X,Y))
            minor = 1/np.sqrt(max(X,Y))

            return major, minor, angle
        else:
            return None, None, None
        
    def assert_valid_ellipse(self, a, b, c):
        if a*c - b**2 > 0:
            return True
        else:
            return False
        
    def calculate_ellipse_rotation_angle(self, a, b, c):
        if a == c:
            angle = 0
        else:
            angle = 0.5*np.arctan((2*b)/(a-c))

        if a > c:
            if b < 0:
                angle = angle-(-0.5*np.pi)
            elif b > 0:
                angle = angle-(0.5*np.pi)

        print(angle,a, b, c)
        return angle
    


    # def assert_diameter(self, semi_axis_x, semi_axis_y):
    #     if semi_axis_x > semi_axis_y:
    #         major, minor = semi_axis_x, semi_axis_y
    #     else:
    #         major, minor = semi_axis_y, semi_axis_x
    #     if (self.major_bound[0] < 2*major < self.major_bound[1]) and (self.minor_bound[0] < 2*minor < self.minor_bound[1]):
    #         flattening = (major - minor) / major
    #         if flattening < self.flattening_bound:
    #             return True
    #     return True
    


    def is_similar(self, p, q, axis1, axis2, angle,accumulator):
        similar_idx = -1
        if accumulator is not None:
            for idx, e in enumerate(accumulator):
                area_dist = np.abs((np.pi*e[2]*e[3] - np.pi * axis1 * axis2))
                center_dist = np.sqrt((e[0] - p)**2 + (e[1] - q)**2)
                angle_dist = (abs(e[4] - angle))
                laxis_dist = abs(max(axis1,axis2)-max(e[2],e[3]))
                saxis_dist = abs(min(axis1,axis2)-min(e[2],e[3]))
                if (laxis_dist < 5) and (center_dist < 5) and ( angle_dist < 0.1745) and(saxis_dist < 10):
                    return idx
        return similar_idx
    


    def average_weight(self, old, now, score):
        return (old * score + now) / (score+1)