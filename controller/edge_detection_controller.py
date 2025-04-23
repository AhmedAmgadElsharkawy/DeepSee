import cv2
import numpy as np
import skimage.exposure as exposure
import utils.utils as utils
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


def prewitt(image,kernel_size,queue = None):
    image = gaussian_filter(image, kernel_size)
    if kernel_size == 3:
        prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    else:
        prewitt_kernel_x = np.array([[2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2]])

        prewitt_kernel_y = np.array([[2, 2, 2, 2, 2],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                        [-1, -1, -1, -1, -1],
                        [-2, -2, -2, -2, -2]])
    prewitt_x = utils.convolution(image, prewitt_kernel_x)
    prewitt_y = utils.convolution(image, prewitt_kernel_y)
    magnitude = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
    prewitt_magnitude = (
        exposure.rescale_intensity(magnitude, in_range="image", out_range=(0, 255))
        .clip(0, 255)
        .astype(np.uint8)
    )

    if queue:
        queue.put(prewitt_magnitude)
    else:
        return prewitt_magnitude

def canny(image,kernel_size,lower_threshold,upper_threshold,variance,queue = None):
    image = gaussian_filter(image, kernel_size, variance)

    edges = cv2.Canny(image, lower_threshold, upper_threshold)

    if queue:
        queue.put(edges)
    else:
        return edges

def canny_handmaded(image,kernel_size,lower_threshold,upper_threshold,variance,queue = None):
    magnitude, angles = sobel(image, variance, kernel_size)
    suppressed = non_maximum_suppression(magnitude, angles)
    edges = hysteresis_thresholding(suppressed, lower_threshold, upper_threshold)
    if queue:
        queue.put(edges)
    else:
        return edges

def sobel(image, variance = 1, kernel_size = None, direction = "Combined",queue = None):

    image = gaussian_filter(image, kernel_size, variance)
    if kernel_size == 3:
        sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    else:
        sobel_kernel_x = np.array([[2, 1, 0, -1, -2],
                        [3, 2, 0, -2, -3],
                        [4, 3, 0, -3, -4],
                        [3, 2, 0, -2, -3],
                        [2, 1, 0, -1, -2]])
        sobel_kernel_y = np.array([[2, 3, 4, 3, 2],
                        [1, 2, 3, 2, 1],
                        [0, 0, 0, 0, 0],
                        [-1, -2, -3, -2, -1],
                        [-2, -3, -4, -3, -2]])
    sobel_x = cv2.filter2D(image, cv2.CV_32F, sobel_kernel_x)
    sobel_y = cv2.filter2D(image, cv2.CV_32F, sobel_kernel_y)

    phase = np.rad2deg(np.arctan2(sobel_y, sobel_x))
    phase[phase < 0] += 180


    if direction == "Horizontal":
        sobel_x_normalized = normalize(sobel_x)
        mag = sobel_x_normalized
    elif direction == "Vertical":
        sobel_y_normalized = normalize(sobel_y)
        mag = sobel_y_normalized
    elif direction == "Combined":
        sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        sobel_magnitude = normalize(sobel_magnitude)
        mag = sobel_magnitude
    
    if queue:
        queue.put(mag)
    else:
        return mag,phase
    
def roberts(image,kernel_size,queue = None):
    image = gaussian_filter(image, kernel_size)

    roberts_x = np.array([[-1, 0], [0, 1]])
    roberts_y = np.array([[0, -1], [1, 0]])

    grad_x = utils.convolution(image, roberts_x)
    grad_y = utils.convolution(image, roberts_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = (
            exposure.rescale_intensity(
                magnitude, in_range="image", out_range=(0, 255)
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
    
    if queue:
        queue.put(magnitude)
    else:
        return magnitude

def normalize(image):
    normalized = (
            exposure.rescale_intensity(
                image, in_range="image", out_range=(0, 255)
            )
            .clip(0, 255)
            .astype(np.uint8)
        )

    return normalized

def non_maximum_suppression(magnitude, angles):
    H, W = magnitude.shape
    suppressed = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angles[i, j] < 67.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            elif 67.5 <= angles[i, j] < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
            elif 112.5 <= angles[i, j] < 157.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0
                
    return suppressed

def hysteresis_thresholding(image, low_threshold, high_threshold):
    rows, cols = image.shape
    strong = 255
    weak = 50

    result = np.zeros((rows, cols), dtype=np.uint8)

    strong_pixels = image >= high_threshold
    weak_pixels = (image >= low_threshold) & (image < high_threshold)

    result[strong_pixels] = strong
    result[weak_pixels] = weak

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if result[i, j] == weak:
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result



class EdgeDetectionWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, image, method, params):
        super().__init__()
        self.image = image
        self.method = method
        self.params = params

    def run(self):
        queue = mp.Queue()
        if self.method == "Sobel Detector":
            target = sobel
            args = (self.image, 1, self.params['kernel_size'], self.params['direction'], queue)
        elif self.method == "Roberts Detector":
            target = roberts
            args = (self.image, self.params['kernel_size'], queue)
        elif self.method == "Prewitt Detector":
            target = prewitt
            args = (self.image, self.params['kernel_size'], queue)
        else:  # Canny (Handmade)
            target = canny_handmaded
            args = (self.image,
                    self.params['kernel_size'],
                    self.params['lower_threshold'],
                    self.params['upper_threshold'],
                    self.params['variance'],
                    queue)

        process = mp.Process(target=target, args=args)
        process.start()

        while True:
            if not queue.empty():
                result = queue.get()
                self.result_ready.emit(result)
                break
            self.msleep(50)

        process.join()


class EdgeDetectionController:
    def __init__(self, edge_detection_window=None):
        self.edge_detection_window = edge_detection_window
        if edge_detection_window:
            edge_detection_window.apply_button.clicked.connect(self.apply_edge_detection)

    def apply_edge_detection(self):
        win = self.edge_detection_window
        image = win.input_image_viewer.image_model.get_gray_image_matrix()
        method = win.edge_detector_type_custom_combo_box.current_text()

        if method == "Sobel Detector":
            params = {
                'kernel_size': win.sobel_detector_kernel_size_spin_box.value(),
                'direction': win.sobel_detector_direction_custom_combo_box.current_text()
            }
        elif method == "Roberts Detector":
            params = {'kernel_size': win.roberts_detector_kernel_size_spin_box.value()}
        elif method == "Prewitt Detector":
            params = {'kernel_size': win.prewitt_detector_kernel_size_spin_box.value()}
        else:
            params = {
                'kernel_size': win.canny_detector_kernel_spin_box.value(),
                'lower_threshold': win.canny_detector_lower_threshold_spin_box.value(),
                'upper_threshold': win.canny_detector_upper_threshold_spin_box.value(),
                'variance': win.canny_detector_variance_spin_box.value()
            }

        win.output_image_viewer.show_loading_effect()
        win.controls_container.setEnabled(False)
        win.image_viewers_container.setEnabled(False)

        self.worker = EdgeDetectionWorker(image, method, params)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()

    def _on_result(self, result_image):
        win = self.edge_detection_window
        win.output_image_viewer.hide_loading_effect()
        win.controls_container.setEnabled(True)
        win.image_viewers_container.setEnabled(True)
        win.output_image_viewer.display_and_set_image_matrix(result_image)
        win.show_toast("Edge Detection is Complete.")
