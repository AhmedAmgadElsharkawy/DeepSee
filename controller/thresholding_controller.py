import numpy as np
import utils.utils as utils
import multiprocessing as mp
from PyQt5.QtCore import QTimer



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


class ThresholdingController():
    def __init__(self,thresholding_window = None):
        self.thresholding_window = thresholding_window
        if self.thresholding_window: 
            self.thresholding_window.apply_button.clicked.connect(self.apply_thresholding)


    def apply_thresholding(self):
        thresholding_type = self.thresholding_window.thresholding_type_custom_combo_box.current_text()
        gray_image = self.thresholding_window.input_image_viewer.image_model.get_gray_image_matrix()

        self.queue = mp.Queue()

        if thresholding_type == "Otsu Thresholding":
            process = mp.Process(target=global_otsu,args=(gray_image,self.queue))
        elif thresholding_type == "Global Mean":
            process = mp.Process(target=global_mean,args=(gray_image,self.queue))
        elif thresholding_type=="Adaptive Mean":
            kernel,constant=self.return_parammeters()
            process = mp.Process(target=adaptive_gaussian_threshold,args=(gray_image,kernel,constant,1,self.queue))
        elif thresholding_type=="Adaptive Gaussian":
            kernel,constant=self.return_parammeters()
            sigma=self.thresholding_window.variance_spin_box.value()
            process = mp.Process(target=adaptive_gaussian_threshold,args=(gray_image,kernel,constant,sigma,self.queue))

        self.thresholding_window.output_image_viewer.show_loading_effect()
        self.thresholding_window.controls_container.setEnabled(False)
        self.thresholding_window.image_viewers_container.setEnabled(False)

        process.start()
        self._start_queue_timer()



    def return_parammeters(self):
        kernel=self.thresholding_window.local_thresholding_window_size_spin_box.value()
        constant=self.thresholding_window.local_thresholding_window_offset_spin_box.value()
        return kernel,constant


    def _start_queue_timer(self):
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self._check_queue)
        self.queue_timer.start(100)

    def _check_queue(self):
        if self.queue and not self.queue.empty():
            self.queue_timer.stop()
            self.thresholding_window.output_image_viewer.hide_loading_effect()
            self.thresholding_window.controls_container.setEnabled(True)
            self.thresholding_window.image_viewers_container.setEnabled(True)
            thresholded_image = self.queue.get()
            self.thresholding_window.output_image_viewer.display_and_set_image_matrix(thresholded_image)
            self.thresholding_window.show_toast(text = "Thresholding is complete.")        
