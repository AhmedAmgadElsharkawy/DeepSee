import cv2
import time
import numpy as np
import multiprocessing as mp
from PyQt5.QtCore import QThread, pyqtSignal


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


def gray_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def draw_corners(image, corners: list) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    corners = np.array(corners).reshape(-1, 2)

    for corner in corners:
        cv2.circle(image, tuple(np.int32(corner[::-1])), 3, (0, 0, 255), -1)

    return image


def harris_corner_detector(image,block_size:int = 2,ksize:int = 3,k:float = 0.04,threshold:float = 0.01,queue = None)->list:
    
    start_time = time.time()
    # Convert image to grayscale if it is colored (3-channel)
    gray = gray_image(image)
        
    
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    if block_size % 2 == 0:
        block_size +=1
    Sxx = cv2.GaussianBlur(Ixx, (block_size, block_size), 1)
    Syy = cv2.GaussianBlur(Iyy, (block_size, block_size), 1)
    Sxy = cv2.GaussianBlur(Ixy, (block_size, block_size), 1)
    
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)
    
    elapsed_time = time.time() - start_time
    print(f"Harris corner detection took {elapsed_time:.4f} seconds")
    
    # Thresholding to get the corners
    corners = np.argwhere(R > threshold * R.max())
    if queue:
        queue.put(draw_corners(image, corners))
    else:
        return draw_corners(image, corners)

def lambda_corner_detector(image, max_corners=10, min_distance=5, quality_level=0.01,queue = None) -> list:
    
    start_time = time.time()
    
    # Convert image to grayscale
    gray = gray_image(image)

    # Compute image gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Sum over window
    window_size = 5

    Sx2 = cv2.GaussianBlur(Ixx, (window_size, window_size), 1.5)
    Sy2 = cv2.GaussianBlur(Iyy, (window_size, window_size), 1.5)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 1.5)

    # Compute minimum eigenvalue (lambda minus) response map
    h, w = gray.shape
    lambda_min = np.zeros((h, w), dtype=np.float32)
                            
    for y in range(h):
                for x in range(w):
                    M = np.array([[Sx2[y, x], Sxy[y, x]],
                                [Sxy[y, x], Sy2[y, x]]])
                    eigvals = np.linalg.eigvalsh(M)  # sorted: [lambda_min, lambda_max]
                    lambda_min[y, x] = eigvals[0]
    

    # Thresholding and NMS
    threshold = quality_level * lambda_min.max()
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(lambda_min, kernel)
    corners = np.column_stack(np.where((lambda_min == dilated) & (lambda_min > threshold)))

    # Sort by strength and apply min_distance
    corners = sorted(corners, key=lambda pt: -lambda_min[pt[0], pt[1]])
    final_corners = []
    for pt in corners:
        if all(np.linalg.norm(pt - fc) >= min_distance for fc in final_corners):
            final_corners.append(pt)
            if len(final_corners) >= max_corners:
                break
    elapsed_time = time.time() - start_time
    print(f"Lambda corner detection took {elapsed_time:.4f} seconds")
    
    if queue:
        queue.put(draw_corners(image, corners))
    else:
        return draw_corners(image, corners)

            
def harris_and_lambda(image,block_size:int = 2,ksize:int = 3,k:float = 0.04,threshold:float = 0.01
                        ,max_corners:int = 10,min_distance:int = 5,quality_level:float = 0.01,queue = None)->list:
    start_time = time.time()
    
    # Convert image to grayscale if it is colored (3-channel)
    gray = gray_image(image)


    # Harris corner detection
    harris_corners = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    harris_corners = cv2.dilate(harris_corners, None)
    # Thresholding to get the corners
    harris_corners_list = np.argwhere(harris_corners > threshold * harris_corners.max())

    # Lambda corner detection
    lambda_corners = cv2.cornerMinEigenVal(gray, blockSize=block_size, ksize=ksize)
    lambda_corners = lambda_corners.astype(np.intp)
    
    harris_corners_list = np.array(harris_corners_list).reshape(-1, 2)
    lambda_corners = np.array(lambda_corners).reshape(-1, 2)

    all_corners = np.concatenate((harris_corners_list, lambda_corners), axis=0)

    elapsed_time = time.time() - start_time
    print(f"Combined Harris + Lambda detection took {elapsed_time:.4f} seconds")

    if queue:
        queue.put(draw_corners(image, all_corners))
    else:
        return draw_corners(image, all_corners)



class ProcessWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, image, method, params):
        super().__init__()
        self.image = image
        self.method = method
        self.params = params

    def run(self):
        queue = mp.Queue()
        if self.method == "Harris Detector":
            target = harris_corner_detector
            args = (self.image,
                    self.params['block_size'],
                    self.params['kernel_size'],
                    self.params['k'],
                    self.params['threshold'],
                    queue)
        elif self.method == "Lambda Detector":
            target = lambda_corner_detector
            args = (self.image,
                    self.params['max_corners'],
                    self.params['min_distance'],
                    self.params['quality_level'],
                    queue)
        else:
            target = harris_and_lambda
            args = (self.image,
                    self.params['block_size'],
                    self.params['kernel_size'],
                    self.params['k'],
                    self.params['threshold'],
                    self.params['max_corners'],
                    self.params['min_distance'],
                    self.params['quality_level'],
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

class CornerDetectionController:
    def __init__(self, corner_detection_window=None):
        self.corner_detection_window = corner_detection_window
        if corner_detection_window:
            corner_detection_window.apply_button.clicked.connect(self.apply_corner_detection)

    def apply_corner_detection(self):
        win = self.corner_detection_window
        image = win.input_image_viewer.image_model.get_image_matrix()
        method = win.corner_detector_type_custom_combo_box.current_text()
        params = {
            'kernel_size': win.harris_detector_kernel_size_spin_box.value(),
            'block_size': win.harris_detector_block_size_spin_box.value(),
            'k': win.harris_detector_k_factor_spin_box.value(),
            'threshold': win.harris_detector_threshold_spin_box.value(),
            'max_corners': win.lambda_detector_max_corners_spin_box.value(),
            'min_distance': win.lambda_detector_min_distance_spin_box.value(),
            'quality_level': win.lambda_detector_quality_level_spin_box.value()
        }
        win.output_image_viewer.show_loading_effect()
        win.controls_container.setEnabled(False)
        win.image_viewers_container.setEnabled(False)

        self.worker = ProcessWorker(image, method, params)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()

    def _on_result(self, output_image):
        win = self.corner_detection_window
        win.output_image_viewer.hide_loading_effect()
        win.controls_container.setEnabled(True)
        win.image_viewers_container.setEnabled(True)
        win.output_image_viewer.display_and_set_image_matrix(output_image)
        win.show_toast("Corner Detection is Complete.")
