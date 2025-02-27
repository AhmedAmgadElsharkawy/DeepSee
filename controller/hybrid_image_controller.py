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
        if image1 is not None:
            type = self.hybrid_image_window.first_image_filter_type_custom_combo_box.current_text()
            if type == "Low Pass Filter":
                img1 = filters.apply_low_pass_filter(gray_image1, radius)
            else:
                img1 = filters.apply_high_pass_filter(gray_image1, radius)
        if image2 is not None:
            type = self.hybrid_image_window.second_image_filter_type_custom_combo_box.current_text()
            if type == "Low Pass Filter":
                img2 = filters.apply_low_pass_filter(gray_image2, radius)
            else:
                img2 = filters.apply_high_pass_filter(gray_image2, radius)