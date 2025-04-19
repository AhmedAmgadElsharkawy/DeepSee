import numpy as np
import utils.utils as utils
import multiprocessing as mp
from PyQt5.QtCore import QTimer



def average_filter(image, kernel_size=3, queue = None):

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    output_image = utils.convolution(image, kernel)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    if queue:
        queue.put(output_image)
    else:
        return output_image


def gaussian_filter(image, kernel_size=3, sigma=1, queue = None):

    # Gaussian kernel
    kernel = utils.gaussian_kernel(kernel_size, sigma)
    output_image = utils.convolution(image, kernel)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    if queue:
        queue.put(output_image)
    else:
        return output_image
    
def median_filter(image, kernel_size=3, queue = None):
    # Pading
    pad_size = kernel_size // 2
    padded_image = utils.pad_image(image=image, pad_size=pad_size)

    # Initialize the output image
    output_image = np.zeros_like(image, dtype=np.float32)


    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output_image[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])


    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    if queue:
        queue.put(output_image)
    else:
        return output_image

def compute_fft(image): # fourier transform
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    return fft_shifted


def compute_ifft(fft_shifted):    # inverse fourier transform
    fft_shifted_back = np.fft.ifftshift(fft_shifted)
    image_filtered = np.fft.ifft2(fft_shifted_back)
    return np.abs(image_filtered)



def create_mask(shape, radius,type):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]        # mesh grid
    if type == "low":
        mask = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2) <= radius
    elif type == "high":
        mask = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2) > radius
    return mask.astype(np.float32)


def apply_low_or_high_pass_filter(image, Radius=2,Type="low", queue = None):
    dft_shift = compute_fft(image)
    mask = create_mask(image.shape, Radius,Type)
    dft_shift_filtered = dft_shift * mask
    output_image = compute_ifft(dft_shift_filtered)
    if queue:
        queue.put(output_image)
    else:
        return dft_shift_filtered, output_image

class FiltersController():
    def __init__(self,filters_window = None):
        self.filters_window = filters_window
        if self.filters_window:
            self.filters_window.apply_button.clicked.connect(self.apply_filter)

    def apply_filter(self):
        filter_type=self.filters_window.filter_type_custom_combo_box.current_text()
        image=self.filters_window.input_image_viewer.image_model.get_gray_image_matrix()

        self.queue = mp.Queue()

        if filter_type == "Average Filter":
            kernel = self.filters_window.average_filter_kernel_size_spin_box.value()
            process =mp.Process(target = average_filter,args=(image, kernel,self.queue))

        elif filter_type == "Gaussian Filter":
            kernel = self.filters_window.guassian_filter_kernel_size_spin_box.value()
            variance=self.filters_window.guassian_filter_variance_spin_box.value()
            process =mp.Process(target = gaussian_filter,args=(image, kernel, variance, self.queue))

        elif filter_type == "Median Filter":
            kernel = self.filters_window.average_filter_kernel_size_spin_box.value()
            process =mp.Process(target = median_filter,args=(image, kernel, self.queue))

        elif filter_type == "Low Pass Filter":
            raduis = self.filters_window.low_pass_filter_radius_spin_box.value()
            process =mp.Process(target = apply_low_or_high_pass_filter,args=(image, raduis,"low" , self.queue))

        elif filter_type == "High Pass Filter":
            raduis = self.filters_window.high_pass_filter_radius_spin_box.value()
            process =mp.Process(target = apply_low_or_high_pass_filter,args=(image, raduis,"high" , self.queue))

        self.filters_window.output_image_viewer.show_loading_effect()
        self.filters_window.controls_container.setEnabled(False)
        self.filters_window.image_viewers_container.setEnabled(False)

        process.start()
        self._start_queue_timer()



    def _start_queue_timer(self):
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self._check_queue)
        self.queue_timer.start(100)

    def _check_queue(self):
        if self.queue and not self.queue.empty():
            self.queue_timer.stop()
            self.filters_window.output_image_viewer.hide_loading_effect()
            self.filters_window.controls_container.setEnabled(True)
            self.filters_window.image_viewers_container.setEnabled(True)
            filtered_img = self.queue.get()
            self.filters_window.output_image_viewer.display_and_set_image_matrix(filtered_img)







