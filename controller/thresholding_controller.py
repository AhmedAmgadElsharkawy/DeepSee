import numpy as np
import utils.utils as utils
class ThresholdingController():
    def __init__(self,thresholding_window):
        self.thresholding_window = thresholding_window
        self.thresholding_window.apply_button.clicked.connect(self.apply_thresholding)


    def apply_thresholding(self):
        thresholding_type = self.thresholding_window.thresholding_type_custom_combo_box.current_text()
        gray_image = self.thresholding_window.input_image_viewer.image_model.get_gray_image_matrix()
        if thresholding_type == "Otsu Thresholding":
            thresholded_image = self.global_otsu(gray_image)
        elif thresholding_type == "Global Mean":
            thresholded_image = self.global_mean(gray_image)
        elif thresholding_type=="Adaptive Mean":
            kernel,constant=self.return_parammeters()
            thresholded_image=self.adaptive_gaussian_threshold(gray_image,kernel_size=kernel,constant=constant)
        elif thresholding_type=="Adaptive Gaussian":
            kernel,constant=self.return_parammeters()
            sigma=self.thresholding_window.variance_spin_box.value()
            thresholded_image=self.adaptive_gaussian_threshold(gray_image,kernel_size=kernel,constant=constant,sigma=sigma)

        self.thresholding_window.output_image_viewer.display_and_set_image_matrix(thresholded_image)

    def global_otsu(self, image):
        hist, bin_edges = np.histogram(image.ravel(), bins=256, range=[0, 256])
        total_pixels = image.size
        hist = hist.astype(np.float32) / total_pixels
        
        cumulative_sum = np.cumsum(hist)
        cumulative_mean = np.cumsum(hist * np.arange(256))
        global_mean = cumulative_mean[-1]

        between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum) + 1e-10)
        optimal_threshold = np.argmax(between_class_variance)

        thresholded_image = (image > optimal_threshold).astype(np.uint8) * 255
        return thresholded_image
    
    def global_mean(self, image):
        T = np.mean(image)
        thresholded = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] >= T:
                    thresholded[i, j] = 255
                else:
                    thresholded[i, j] = 0

        return thresholded

    def return_parammeters(self):
            kernel=self.thresholding_window.local_thresholding_window_size_spin_box.value()
            constant=self.thresholding_window.local_thresholding_window_offset_spin_box.value()
            return kernel,constant
    def adaptive_mean_threshold(self, image, kernel_size=11, constant=2):

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



    def adaptive_gaussian_threshold(self, image, kernel_size=11, constant=2,sigma=1):

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


