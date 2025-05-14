import numpy as np
import math
import cv2
import multiprocessing as mp
from PyQt5.QtCore import QThread, pyqtSignal

from controller.filters_controller import gaussian_filter


"""
    we use multiprocessing to run the algorithms on seperate core from the gui

    The creation of the process take some time which freeezes the gui if run on the same therad so 
    use multithreading to do the logic of creating and tracking the process on different thread

    Create IPC queue to share resources between cores
    IPC stands for Inter‑Process Communication. 
    It’s the set of mechanisms an operating system (and your programs) provide 
    to let separate processes—each with its own private memory space—exchange data and signals.
    Without IPC, one process couldn’t directly read or write another process’s memory.

    any function you pass to multiprocessing.Process must be defined at the top level of a module, not 
    inside a class or another function. That way, Python’s pickler (serialize and deserialize data between processes memory) can import it by name

    not all objects can be easily pickled. For example:
    Functions or classes defined inside another function: These can't be pickled because they don't have a top-level name, and pickling requires that objects can be imported by name.
    Local variables: Variables that are only local to a function or method might also pose issues with pickling.  

    underscore before a function or variable name in Python is a convention that 
    This is a private/internal function or variable. Please don’t use it outside this class/module unless you really know what you’re doing. 
"""


def process_active_conntour(input_image,image,num_iterations,radius,num_points,window_size,alpha,beta,gamma,queue = None):
    image = gaussian_filter(image, kernel_size=window_size, sigma=1)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    curve = initialize_contours(center, radius, num_points)
    output_image = np.zeros_like(input_image)
    for i in range(num_iterations):
        new_curve=snake_operation(image, curve, window_size, alpha, beta, gamma)
    curve=new_curve

    output_image,contour_area,contour_perimeter,chain_code=process_contour(input_image, output_image, curve)
    if queue:
        queue.put((output_image,contour_area,contour_perimeter,chain_code))
    else:
        return output_image,contour_area,contour_perimeter,chain_code

def initialize_contours(center, radius, number_of_points):
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

def calculate_internal_energy(point, prev_point, next_point, alpha, beta, avg_distance):
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
def calculate_external_energy(point, grad_x, grad_y, gamma):
    x, y = point
    if 0 <= y < grad_x.shape[0] and 0 <= x < grad_x.shape[1]:
        gx = grad_x[y, x]
        gy = grad_y[y, x]
        return -gamma * (gx ** 2 + gy ** 2)
    return 0.0


def calculate_total_energy(point, prev_point, next_point, alpha, beta, gamma, grad_x, grad_y,avg_distance):
    internal = calculate_internal_energy(point, prev_point, next_point, alpha, beta,avg_distance)
    external = calculate_external_energy(point, grad_x, grad_y, gamma)
    return internal + external

def snake_operation(image, curve, window_size, alpha, beta, gamma):
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
        avg_distance = calculate_average_distance(curve)

        for dx in range(-window_index, window_index + 1):
            for dy in range(-window_index, window_index + 1):
                moved_pt = (pt[0] + dx, pt[1] + dy)
                if 0 <= moved_pt[0] < image.shape[1] and 0 <= moved_pt[1] < image.shape[0]:
                    energy = calculate_total_energy( moved_pt, prev_pt, next_pt, alpha, beta, gamma,grad_x,grad_y,avg_distance)
                    if energy < min_energy:
                        min_energy = energy
                        best_pt = moved_pt

        new_curve[i] = best_pt

    curve[:] = new_curve
    return curve

def process_contour(image, output_image, snake_points):
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
        direction_code = get_chain_code_direction(dx,dy)
        chain_code.append(direction_code)

        j = i  # Update j for next iteration

    area = abs(area / 2.0)

    return output_image, area, perimeter,chain_code


def get_chain_code_direction(dx, dy):
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

def calculate_average_distance(curve):
    distances = [
        np.hypot(curve[(i + 1) % len(curve)][0] - curve[i][0],
                    curve[(i + 1) % len(curve)][1] - curve[i][1])
        for i in range(len(curve))
    ]
    return np.mean(distances)



class ActiveContourWorker(QThread):
    result_ready = pyqtSignal(np.ndarray, float, float, list)

    def __init__(self, input_image, image, num_iterations, radius, num_points, window_size, alpha, beta, gamma):
        super().__init__()
        self.input_image = input_image
        self.image = image
        self.num_iterations = num_iterations
        self.radius = radius
        self.num_points = num_points
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def run(self):
        queue = mp.Queue()
        process = mp.Process(
            target=process_active_conntour,
            args=(self.input_image, self.image, self.num_iterations, self.radius,
                  self.num_points, self.window_size, self.alpha, self.beta, self.gamma, queue)
        )
        process.start()

        while True:
            if not queue.empty():
                output_image, contour_area, contour_perimeter, chain_code = queue.get()
                self.result_ready.emit(output_image, contour_area, contour_perimeter, chain_code)
                break
            self.msleep(50)

        process.join()




class ActiveContoursController:
    def __init__(self, active_contours_window=None):
        self.active_contours_window = active_contours_window
        if active_contours_window:
            active_contours_window.apply_button.clicked.connect(self.apply_active_contour)

    def apply_active_contour(self):
        win = self.active_contours_window
        input_image = win.input_image_viewer.image_model.get_image_matrix()
        image = win.input_image_viewer.image_model.get_gray_image_matrix()

        params = {
            'num_iterations': win.active_contours_iterations_spin_box.value(),
            'radius': win.active_contours_radius_spin_box.value(),
            'num_points': win.active_contours_points_spin_box.value(),
            'window_size': win.active_contours_window_size_spin_box.value(),
            'alpha': win.active_contours_detector_alpha_spin_box.value(),
            'beta': win.active_contours_detector_beta_spin_box.value(),
            'gamma': win.active_contours_detector_gamma_spin_box.value(),
        }

        win.output_image_viewer.show_loading_effect()
        win.controls_container.setEnabled(False)
        win.image_viewers_container.setEnabled(False)

        self.worker = ActiveContourWorker(
            input_image, image,
            params['num_iterations'], params['radius'], params['num_points'],
            params['window_size'], params['alpha'], params['beta'], params['gamma']
        )
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()

    def _on_result(self, output_image, contour_area, contour_perimeter, chain_code):
        win = self.active_contours_window
        win.output_image_viewer.hide_loading_effect()
        win.controls_container.setEnabled(True)
        win.image_viewers_container.setEnabled(True)
        win.output_image_viewer.display_and_set_image_matrix(output_image)
        self.update_perimeter_area(contour_perimeter, contour_area)
        self.update_chain_code_display(chain_code)
        win.show_toast("Active Contour is Complete.")

    def update_perimeter_area(self, contour_perimeter, contour_area):
        self.active_contours_window.active_contours_detector_perimeter.setText(f"{contour_perimeter:.2f}")
        self.active_contours_window.active_contours_detector_area.setText(f"{contour_area:.2f}")

    def update_chain_code_display(self, chain_code):
        chain_text = "".join(str(code) for code in chain_code)
        formatted_text = self.format_chain_code(chain_text)
        self.active_contours_window.active_contours_detector_chaincode.setText(f"{formatted_text}")

    def format_chain_code(self, chain_code: str, line_width: int = 40) -> str:
        return '\n'.join(chain_code[i:i + line_width] for i in range(0, len(chain_code), line_width))















