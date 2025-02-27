import numpy as np
class ThresholdingController():
    def __init__(self,thresholding_window):
        self.thresholding_window = thresholding_window
        self.thresholding_window.apply_button.clicked.connect(self.apply_thresholding)

    def apply_thresholding(self):
        gray_image = self.thresholding_window.input_image_viewer.image_model.get_gray_image_matrix()
        thresholded_image = self.global_otsu(gray_image)
        self.thresholding_window.output_image_viewer.display_and_set_image_matrix(thresholded_image)

    def global_otsu(self, image):
        hist, bin_edges = np.histogram(image.ravel(), bins=256, range=[0, 256])
        total_pixels = image.size
        hist = hist.astype(np.float32) / total_pixels  # Normalize
        
        cumulative_sum = np.cumsum(hist)
        cumulative_mean = np.cumsum(hist * np.arange(256))
        global_mean = cumulative_mean[-1]

        between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum) + 1e-10)
        optimal_threshold = np.argmax(between_class_variance)

        thresholded_image = (image > optimal_threshold).astype(np.uint8) * 255
        return thresholded_image