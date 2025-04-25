import numpy as np
import utils.utils as utils
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


def thresholding_process(thresholding_type,thresholding_scope,params,queue):
    if thresholding_scope == "Global Thresholding":
        thresholded_image = apply_global_thresholding(thresholding_type,params["gray_image"])
    else:
        thresholded_image = apply_local_thresholding(thresholding_type,image = params["gray_image"],window_size = params["window_size"],offset = params["offset"],sigma = params['sigma'])
    queue.put(thresholded_image)


def apply_global_thresholding(thresholding_type,image):
    if thresholding_type == "Global Mean":
        thresholded_image = global_mean(image)
    elif thresholding_type == "Otsu Thresholding":
        thresholded_image, _ = otsu_thresholding(image)
    elif thresholding_type == "Optimal Thresholding":
        thresholded_image,_ = optimal_thresholding(image)

    return thresholded_image

def apply_local_thresholding(thresholding_type, image, window_size, offset, sigma):
    if thresholding_type=="Adaptive Mean":
        return adaptive_mean_threshold (image, window_size, offset)
    elif thresholding_type=="Adaptive Gaussian":
        return  adaptive_gaussian_threshold(image, window_size, offset,sigma)

    thresholded_image = np.zeros(image.shape[:2], dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    window = (window_size, window_size)
    step_y, step_x = window

    for y in range(0, image_height, step_y):
        for x in range(0, image_width, step_x):

            sub_image = image[y : min(y + step_y, image_height), x : min(x + step_x, image_width)]

            if thresholding_type == "Optimal Thresholding":
                _, threshold = optimal_thresholding(sub_image)
                threshold = threshold - offset
                local_thresholded = (sub_image > threshold).astype(np.uint8) * 255

            if thresholding_type == "Otsu Thresholding":
                _, threshold = otsu_thresholding(sub_image)
                threshold = threshold - offset
                local_thresholded = (sub_image > threshold).astype(np.uint8) * 255
            
            thresholded_image[y : min(y + step_y, image_height), x : min(x + step_x, image_width)] = local_thresholded

    return thresholded_image

def optimal_thresholding(image):
    height, width = image.shape[:2]
    corners = [
        image[0, 0],
        image[0, width - 1],
        image[height - 1, 0],
        image[height - 1, width - 1],
    ]
    threshold = np.mean(corners)
    while True:
        class1_mean = np.mean(image[image < threshold])
        class2_mean = np.mean(image[image >= threshold])
        new_threshold = (class1_mean + class2_mean) / 2
        if np.abs(new_threshold - threshold) < 1e-6:
            break
        threshold = new_threshold

    thresholded_image = (image > threshold).astype(np.uint8) * 255

    return thresholded_image, threshold

def otsu_thresholding(image):
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=[0, 256])
    total_pixels = image.size
    hist = hist.astype(np.float32) / total_pixels
    
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumulative_mean[-1]

    between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum) + 1e-10)
    optimal_threshold = np.argmax(between_class_variance)

    thresholded_image = (image > optimal_threshold).astype(np.uint8) * 255
    
    return thresholded_image, optimal_threshold

def global_mean(image):
    T = np.mean(image)
    thresholded = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= T:
                thresholded[i, j] = 255
            else:
                thresholded[i, j] = 0

    return thresholded

def adaptive_mean_threshold(image, kernel_size=11, constant=2):

    height, width = image.shape

    output_image = np.zeros_like(image)

    pad_size = kernel_size // 2
    # Pading
    padded_image = utils.pad_image(image=image, pad_size=pad_size, pading_type="reflect")
    for y in range(height):
        for x in range(width):

            neighborhood = padded_image[y:y + kernel_size, x:x + kernel_size]

            mean_value = np.mean(neighborhood)

            # thresholding
            if image[y, x] > (mean_value - constant):
                output_image[y, x] = 255
            else:
                output_image[y, x] = 0

    return output_image

def adaptive_gaussian_threshold(image, kernel_size=11, constant=2,sigma=1):

    height, width = image.shape

    output_image = np.zeros_like(image)
    pad_size = kernel_size // 2
    kernel=utils.gaussian_kernel(kernel_size,sigma=sigma)

    padded_image = utils.pad_image(image=image, pad_size=pad_size, pading_type="reflect")

    for y in range(height):
        for x in range(width):
            neighborhood = padded_image[y:y + kernel_size, x:x + kernel_size]
            mean_value = np.sum(neighborhood*kernel)
            if image[y, x] > (mean_value - constant):
                output_image[y, x] = 255
            else:
                output_image[y, x] = 0

    return output_image
    
class ThresholdingProcessWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self,thresholding_type,thresholding_scope,params):
        super().__init__()
        self.params = params
        self.thresholding_type = thresholding_type
        self.thresholding_scope = thresholding_scope

    def run(self):  
        queue = mp.Queue()

        process = mp.Process(target = thresholding_process,args=(self.thresholding_type,self.thresholding_scope,self.params,queue))

        process.start()
        while True:
            if not queue.empty():
                result = queue.get()
                self.result_ready.emit(result)
                break
            self.msleep(50)
        process.join()

class ThresholdingController():
    def __init__(self,thresholding_window = None):
        self.thresholding_window = thresholding_window
        if self.thresholding_window: 
            self.thresholding_window.apply_button.clicked.connect(self.apply_thresholding)

    def apply_thresholding(self):
        params = {}
        thresholding_scope = self.thresholding_window.thresholding_scope_custom_combo_box.current_text()
        thresholding_type = self.thresholding_window.thresholding_type_custom_combo_box.current_text()
        params['gray_image'] = self.thresholding_window.input_image_viewer.image_model.get_gray_image_matrix()
        params['window_size']=self.thresholding_window.local_thresholding_window_size_spin_box.value()
        params['offset']=self.thresholding_window.local_thresholding_window_offset_spin_box.value()
        params['sigma']=self.thresholding_window.variance_spin_box.value()

        self.thresholding_window.output_image_viewer.show_loading_effect()
        self.thresholding_window.controls_container.setEnabled(False)
        self.thresholding_window.image_viewers_container.setEnabled(False)


        self.worker = ThresholdingProcessWorker(thresholding_type,thresholding_scope,params)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()
    
    def _on_result(self,result):
        self.thresholding_window.output_image_viewer.hide_loading_effect()
        self.thresholding_window.controls_container.setEnabled(True)
        self.thresholding_window.image_viewers_container.setEnabled(True)
        self.thresholding_window.output_image_viewer.display_and_set_image_matrix(result)
        self.thresholding_window.show_toast(text = "Thresholding is complete.")  