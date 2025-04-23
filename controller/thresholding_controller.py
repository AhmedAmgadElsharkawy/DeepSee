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



def global_otsu(image, queue = None):
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=[0, 256])
    total_pixels = image.size
    hist = hist.astype(np.float32) / total_pixels
    
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumulative_mean[-1]

    between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum) + 1e-10)
    optimal_threshold = np.argmax(between_class_variance)

    thresholded_image = (image > optimal_threshold).astype(np.uint8) * 255
    if queue:
        queue.put(thresholded_image)
    else:    
        return thresholded_image

def global_mean(image, queue = None):
    T = np.mean(image)
    thresholded = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= T:
                thresholded[i, j] = 255
            else:
                thresholded[i, j] = 0

    if queue:
        queue.put(thresholded)
    else:    
        return thresholded


def adaptive_mean_threshold(image, kernel_size=11, constant=2, queue = None):

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

    if queue:
        queue.put(output_image)
    else:    
        return output_image



def adaptive_gaussian_threshold(image, kernel_size=11, constant=2,sigma=1, queue = None):

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

    if queue:
        queue.put(output_image)
    else:    
        return output_image
    

class ThresholdingProcessWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self,thresholding_type,params):
        super().__init__()
        self.params = params
        self.thresholding_type = thresholding_type

    def run(self):  
        queue = mp.Queue()

        if self.thresholding_type == "Otsu Thresholding":
            process = mp.Process(target=global_otsu,args=(self.params['gray_image'],queue))
        elif self.thresholding_type == "Global Mean":
            process = mp.Process(target=global_mean,args=(self.params['gray_image'],queue))
        elif self.thresholding_type=="Adaptive Mean":
            process = mp.Process(target=adaptive_gaussian_threshold,args=(self.params['gray_image'],self.params['kernel'],self.params['constant'],1,queue))
        elif self.thresholding_type=="Adaptive Gaussian":
            process = mp.Process(target=adaptive_gaussian_threshold,args=(self.params['gray_image'],self.params['kernel'],self.params['constant'],self.params['sigma'],queue))

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
        thresholding_type = self.thresholding_window.thresholding_type_custom_combo_box.current_text()
        params['gray_image'] = self.thresholding_window.input_image_viewer.image_model.get_gray_image_matrix()
        
        if thresholding_type=="Adaptive Mean":
            params['kernel'],params['constant']=self.return_parammeters()
        elif thresholding_type=="Adaptive Gaussian":
            params['kernel'],params['constant']=self.return_parammeters()
            params['sigma']=self.thresholding_window.variance_spin_box.value()

        self.thresholding_window.output_image_viewer.show_loading_effect()
        self.thresholding_window.controls_container.setEnabled(False)
        self.thresholding_window.image_viewers_container.setEnabled(False)


        self.worker = ThresholdingProcessWorker(thresholding_type,params)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()



    def return_parammeters(self):
        kernel=self.thresholding_window.local_thresholding_window_size_spin_box.value()
        constant=self.thresholding_window.local_thresholding_window_offset_spin_box.value()
        return kernel,constant

    
    def _on_result(self,result):
        self.thresholding_window.output_image_viewer.hide_loading_effect()
        self.thresholding_window.controls_container.setEnabled(True)
        self.thresholding_window.image_viewers_container.setEnabled(True)
        self.thresholding_window.output_image_viewer.display_and_set_image_matrix(result)
        self.thresholding_window.show_toast(text = "Thresholding is complete.")  