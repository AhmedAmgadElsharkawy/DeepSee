import numpy as np
class ThresholdingController():
    def __init__(self,thresholding_window):
        self.thresholding_window = thresholding_window
        self.thresholding_window.apply_button.clicked.connect(self.apply_thresholding)


    def apply_thresholding(self):
        filter_type = self.thresholding_window.thresholding_type_custom_combo_box.current_text()
        gray_image = self.thresholding_window.input_image_viewer.image_model.get_gray_image_matrix()
        if filter_type == "Otsu Thresholding":
            thresholded_image = self.global_otsu(gray_image)
        elif filter_type == "Global Mean":
            thresholded_image = self.global_mean(gray_image)
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