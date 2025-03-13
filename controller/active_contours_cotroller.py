import numpy as np
import math
import cv2
class ActiveContoursController():
    def __init__(self,active_contours_window):
        self.active_contours_window = active_contours_window
        self.active_contours_window.apply_button.clicked.connect(self.apply_active_contour)


    def apply_active_contour(self):
        input_image = self.active_contours_window.input_image_viewer.image_model.get_image_matrix()
        image=self.active_contours_window.input_image_viewer.image_model.get_gray_image_matrix()
        num_iterations=self.active_contours_window.active_contours_iterations_spin_box.value()
        radius=self.active_contours_window.active_contours_radius_spin_box.value()
        num_points=self.active_contours_window.active_contours_points_spin_box.value()
        window_size=self.active_contours_window.active_contours_window_size_spin_box.value()
        alpha=self.active_contours_window.active_contours_detector_alpha_spin_box.value()
        beta=self.active_contours_window.active_contours_detector_beta_spin_box.value()
        gamma=self.active_contours_window.active_contours_detector_gamma_spin_box.value()
        filters = self.active_contours_window.main_window.filters_window.filters_controller
        image=filters.gaussian_filter(image, kernel_size=window_size, sigma=1)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        curve = self.initialize_contours(center, radius, num_points)
        output_image = np.zeros_like(input_image)
        for i in range(num_iterations):
            new_curve=self.snake_operation(image, curve, window_size, alpha, beta, gamma)
        curve=new_curve
        output_image=self.draw_contours(input_image, output_image, curve)
        self.active_contours_window.output_image_viewer.display_and_set_image_matrix(output_image)
        print("apply_active_contour")


    def initialize_contours(self,center, radius, number_of_points):
        curve = []
        current_angle = 0
        resolution = 360 / number_of_points

        for i in range(number_of_points):
            angle = np.deg2rad(current_angle)
            x = int(radius * np.cos(angle) + center[0])
            y = int(radius * np.sin(angle) + center[1])
            current_angle += resolution
            curve.append((x, y))
        return curve
    def points_distance(self,x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    def calculate_internal_energy(self,point, previous_point, next_point, alpha):
        dx1 = point[0] - previous_point[0]
        dy1 = point[1] - previous_point[1]
        dx2 = next_point[0] - point[0]
        dy2 = next_point[1] - point[1]
        denominator = (dx1 ** 2 + dy1 ** 2) ** 1.5

        # Handle division by zero
        if denominator == 0:
            return 0.0  # Return zero curvature if denominator is zero

        curvature = (dx1 * dy2 - dx2 * dy1) / denominator
        return alpha * curvature
    def calculate_external_energy(self,image, point, beta):
        # Ensure the point is within the image bounds
        if 0 <= point[1] < image.shape[0] and 0 <= point[0] < image.shape[1]:
            return -beta * image[point[1], point[0]]
        else:
            return 0.0  # Return zero energy if the point is outside the image

    def calculate_gradients(self,point, prev_point, gamma):
        dx = point[0] - prev_point[0]
        dy = point[1] - prev_point[1]
        return gamma * (dx ** 2 + dy ** 2)
    def calculate_point_energy(self,image, point, prev_point, next_point, alpha, beta, gamma):
        internal_energy = self.calculate_internal_energy(point, prev_point, next_point, alpha)
        external_energy = self.calculate_external_energy(image, point, beta)
        gradients = self.calculate_gradients(point, prev_point, gamma)
        return internal_energy + external_energy + gradients

    def snake_operation(self,image, curve, window_size, alpha, beta, gamma):
        window_index = (window_size - 1) // 2
        num_points = len(curve)
        new_curve = [None] * num_points

        for i in range(num_points):
            pt = curve[i]
            prev_pt = curve[(i - 1 + num_points) % num_points]
            next_pt = curve[(i + 1) % num_points]
            min_energy = float('inf')
            new_pt = pt

            for dx in range(-window_index, window_index + 1):
                for dy in range(-window_index, window_index + 1):
                    move_pt = (pt[0] + dx, pt[1] + dy)
                    energy = self.calculate_point_energy(image, move_pt, prev_pt, next_pt, alpha, beta, gamma)
                    if energy < min_energy:
                        min_energy = energy
                        new_pt = move_pt

            new_curve[i] = new_pt

        curve[:] = new_curve
        return curve

    def draw_contours(self,image, output_image, snake_points, chain_code=None):
        output_image[:] = image.copy()
        # chain_code.clear()

        for i in range(len(snake_points)):
            cv2.circle(output_image, snake_points[i], 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(output_image, snake_points[i - 1], snake_points[i], (255, 0, 0), 2)
            cv2.line(output_image, snake_points[0], snake_points[-1], (255, 0, 0), 2)

            # dx = snake_points[(i + 1) % len(snake_points)][0] - snake_points[i][0]
            # dy = snake_points[(i + 1) % len(snake_points)][1] - snake_points[i][1]
            #
            # code = None
            # if dx == 0 and dy == 1:
            #     code = 0
            # elif dx == 1 and dy == 1:
            #     code = 1
            # elif dx == 1 and dy == 0:
            #     code = 2
            # elif dx == 1 and dy == -1:
            #     code = 3
            # elif dx == 0 and dy == -1:
            #     code = 4
            # elif dx == -1 and dy == -1:
            #     code = 5
            # elif dx == -1 and dy == 0:
            #     code = 6
            # elif dx == -1 and dy == 1:
            #     code = 7
            #
            # chain_code.append(code)
        return output_image





