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
        output_image,contour_area,contour_perimeter,chain_code=self.process_contour(input_image, output_image, curve)
        self.update_perimeter_area(contour_perimeter,contour_area )
        self.update_chain_code_display(chain_code)
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

    def calculate_internal_energy(self,point, prev_point, next_point, alpha, beta, avg_distance):
        # Elasticity with penalty
        dx1 = point[0] - prev_point[0]
        dy1 = point[1] - prev_point[1]
        dist1 = np.hypot(dx1, dy1)
        elasticity = alpha * ((dist1 - avg_distance) ** 2)

        # Curvature term
        dx2 = next_point[0] - 2 * point[0] + prev_point[0]
        dy2 = next_point[1] - 2 * point[1] + prev_point[1]
        curvature = beta * (dx2 ** 2 + dy2 ** 2)

        return elasticity + curvature
    def calculate_external_energy(self,point, grad_x, grad_y, gamma):
        x, y = point
        if 0 <= y < grad_x.shape[0] and 0 <= x < grad_x.shape[1]:
            gx = grad_x[y, x]
            gy = grad_y[y, x]
            return -gamma * (gx ** 2 + gy ** 2)
        return 0.0


    def calculate_total_energy(self,point, prev_point, next_point, alpha, beta, gamma, grad_x, grad_y,avg_distance):
        internal = self.calculate_internal_energy(point, prev_point, next_point, alpha, beta,avg_distance)
        external = self.calculate_external_energy(point, grad_x, grad_y, gamma)
        return internal + external

    def snake_operation(self,image, curve, window_size, alpha, beta, gamma):
        window_index = (window_size - 1) // 2
        num_points = len(curve)
        new_curve = [None] * num_points
        grad_y, grad_x = np.gradient(image.astype(float))

        for i in range(num_points):
            pt = curve[i]
            prev_pt = curve[(i - 1 + num_points) % num_points]
            next_pt = curve[(i + 1) % num_points]
            min_energy = float('inf')
            best_pt = pt
            avg_distance = self.calculate_average_distance(curve)

            for dx in range(-window_index, window_index + 1):
                for dy in range(-window_index, window_index + 1):
                    moved_pt = (pt[0] + dx, pt[1] + dy)
                    if 0 <= moved_pt[0] < image.shape[1] and 0 <= moved_pt[1] < image.shape[0]:
                        energy = self.calculate_total_energy( moved_pt, prev_pt, next_pt, alpha, beta, gamma,grad_x,grad_y,avg_distance)
                        if energy < min_energy:
                            min_energy = energy
                            best_pt = moved_pt

            new_curve[i] = best_pt

        curve[:] = new_curve
        return curve

    def process_contour(self, image, output_image, snake_points):
        output_image[:] = image.copy()
        area = 0.0
        perimeter = 0.0
        chain_code = []
        j = len(snake_points) - 1

        for i in range(len(snake_points)):
            # Draw points
            cv2.circle(output_image, snake_points[i], 4, (0, 0, 255), -1)

            # Draw lines
            if i > 0:
                cv2.line(output_image, snake_points[i - 1], snake_points[i], (255, 0, 0), 2)
            # Closing the contour loop
            if i == len(snake_points) - 1:
                cv2.line(output_image, snake_points[i], snake_points[0], (255, 0, 0), 2)

            # Area calculation (Shoelace formula)
            area += (snake_points[i][0] * snake_points[j][1]) - (snake_points[j][0] * snake_points[i][1]);


            # Perimeter calculation
            next_i = (i + 1) % len(snake_points)
            dx = snake_points[i][0] - snake_points[next_i][0]
            dy = snake_points[i][1] - snake_points[next_i][1]

            perimeter += math.hypot(dx, dy)
            direction_code = self.get_chain_code_direction(dx,dy)
            chain_code.append(direction_code)
            print(direction_code)

            j = i  # Update j for next iteration

        area = abs(area / 2.0)

        return output_image, area, perimeter,chain_code


    def get_chain_code_direction(self,dx, dy):
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        if angle_deg < 0:
            angle_deg += 360
        if angle_deg >= 337.5 or angle_deg < 22.5:
            return 0
        elif angle_deg < 67.5:
            return 1
        elif angle_deg < 112.5:
            return 2
        elif angle_deg < 157.5:
            return 3
        elif angle_deg < 202.5:
            return 4
        elif angle_deg < 247.5:
            return 5
        elif angle_deg < 292.5:
            return 6
        else:
            return 7

    def calculate_average_distance(self,curve):
        distances = [
            np.hypot(curve[(i + 1) % len(curve)][0] - curve[i][0],
                     curve[(i + 1) % len(curve)][1] - curve[i][1])
            for i in range(len(curve))
        ]
        return np.mean(distances)


    def update_perimeter_area(self, contour_perimeter, contour_area ):
        self.active_contours_window.active_contours_detector_perimeter.setText(f"{contour_perimeter:.2f} ")
        self.active_contours_window.active_contours_detector_area.setText(f"{contour_area:.2f} ")

    def update_chain_code_display(self, chain_code):
        chain_text = "".join(str(code) for code in chain_code)
        chain_text=self.format_chain_code(chain_text)
        print(chain_text)
        self.active_contours_window.active_contours_detector_chaincode.setText(f"{chain_text} ")
        # self.active_contours_window.active_contours_detector_chaincode.setPlainText(f"{chain_text} ")

    def format_chain_code(self,chain_code: str, line_width: int = 40) -> str:
        return '\n'.join(chain_code[i:i+line_width] for i in range(0, len(chain_code), line_width))












