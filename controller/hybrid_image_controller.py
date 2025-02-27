import cv2
class HybridImageController():
    def __init__(self,hybrid_image_window):
        self.hybrid_image_window = hybrid_image_window
        self.hybrid_image_window.apply_button.clicked.connect(self.apply_image_mixing)

    def apply_image_mixing(self):
        filters = self.hybrid_image_window.main_window.filters_window.filters_controller
        image1 = self.hybrid_image_window.first_original_image_viewer.image_model.get_image_matrix()
        gray_image1 = self.hybrid_image_window.first_original_image_viewer.image_model.get_gray_image_matrix()
        image2 = self.hybrid_image_window.second_original_image_viewer.image_model.get_image_matrix()
        gray_image2 = self.hybrid_image_window.second_original_image_viewer.image_model.get_gray_image_matrix()
        radius = self.hybrid_image_window.radius_custom_spin_box.value()
        if gray_image1 is not None and gray_image2 is not None:
            gray_image1, gray_image2 = self.img_adjustment(gray_image1, gray_image2)
        result = None

        if image1 is not None:
            type1 = self.hybrid_image_window.first_image_filter_type_custom_combo_box.current_text()
            if type1 == "Low Pass Filter":
                fft1, img1 = filters.apply_low_pass_filter(gray_image1, radius)
            else:
                fft1, img1 = filters.apply_high_pass_filter(gray_image1, radius)
            result = fft1
        if image2 is not None:
            type2 = self.hybrid_image_window.second_image_filter_type_custom_combo_box.current_text()
            if type2 == "Low Pass Filter":
                fft2, img2 = filters.apply_low_pass_filter(gray_image2, radius)
            else:
                fft2, img2 = filters.apply_high_pass_filter(gray_image2, radius)
            if result is not None:
                result += fft2
            else:
                result = fft2

        if gray_image1 is not None:
            self.hybrid_image_window.first_filtered_image_viewer.display_image_matrix2(img1)

        if gray_image2 is not None:
            self.hybrid_image_window.second_filtered_image_viewer.display_image_matrix2(img2)

        if result is not None:
            result = filters.compute_ifft(result)
            self.hybrid_image_window.hybrid_image_viewer.display_image_matrix2(result)

    def img_adjustment(self, image_1, image_2):
        width_1, height_1 = image_1.shape
        width_2, height_2 = image_2.shape
        if width_1 > width_2:
            width = width_2
        else:
            width = width_1

        if height_1 > height_2:
            height = height_2
        else:
            height = height_1
        adjusted_img1 = cv2.resize(image_1, (width, height))
        adjusted_img2 = cv2.resize(image_2, (width, height))
        return adjusted_img1, adjusted_img2

        