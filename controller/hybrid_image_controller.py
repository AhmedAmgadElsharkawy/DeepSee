import cv2
import multiprocessing as mp
from PyQt5.QtCore import QTimer

from controller.filters_controller import compute_ifft,apply_low_or_high_pass_filter


def hybrid_image_process(image1,image2,radius,type1,type2,queue):
    if image1 is not None and image2 is not None:
        image1, image2 = img_adjustment(image1, image2)
        
    img1, img2, mix = calculate_ffts(image1, image2, radius,type1,type2)

    inverse_mix = compute_ifft(mix)

    if queue:
        queue.put((img1,img2,mix,inverse_mix))
    else:
        return img1,img2,mix,inverse_mix

def img_adjustment(image_1, image_2):
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
    adjusted_img1 = cv2.resize(image_1, (height, width))
    adjusted_img2 = cv2.resize(image_2, (height, width))
    return adjusted_img1, adjusted_img2

def calculate_ffts(image1, image2,radius,type1,type2):
    result, img1, img2 = None, None, None
    
    if image1 is not None:
        if type1 == "Low Pass Filter":
            fft1, img1 = apply_low_or_high_pass_filter(image1, radius,"low")
        else:
            fft1, img1 = apply_low_or_high_pass_filter(image1, radius,"high")
        result = fft1
    if image2 is not None:
        if type2 == "Low Pass Filter":
            fft2, img2 = apply_low_or_high_pass_filter(image2, radius,"low")
        else:
            fft2, img2 = apply_low_or_high_pass_filter(image2, radius,"high")
        if result is not None:
            result += fft2
        else:
            result = fft2

    return img1, img2, result


class HybridImageController():
    def __init__(self,hybrid_image_window = None):
        self.hybrid_image_window = hybrid_image_window
        if self.hybrid_image_window:
            self.hybrid_image_window.apply_button.clicked.connect(self.apply_image_mixing)
            self.hybrid_image_window.first_image_filter_type_custom_combo_box.currentIndexChanged.connect(
                lambda value: self.update_other_combo(self.hybrid_image_window.second_image_filter_type_custom_combo_box, value)
            )
            self.hybrid_image_window.second_image_filter_type_custom_combo_box.currentIndexChanged.connect(
                lambda value: self.update_other_combo(self.hybrid_image_window.first_image_filter_type_custom_combo_box, value)
            )

    def update_other_combo(self, combo_box, value):
        combo_box.combo_box.blockSignals(True)
        combo_box.combo_box.setCurrentIndex(1 - value)
        combo_box.combo_box.blockSignals(False)

    def apply_image_mixing(self):
        image1 = self.hybrid_image_window.first_original_image_viewer.image_model.get_gray_image_matrix()
        image2 = self.hybrid_image_window.second_original_image_viewer.image_model.get_gray_image_matrix()
        radius = self.hybrid_image_window.radius_custom_spin_box.value()
        type1 = self.hybrid_image_window.first_image_filter_type_custom_combo_box.current_text()
        type2 = self.hybrid_image_window.second_image_filter_type_custom_combo_box.current_text()
        
        self.queue = mp.Queue()

        self.hybrid_image_window.hybrid_image_viewer.show_loading_effect()
        if image1 is not None:
            self.hybrid_image_window.first_filtered_image_viewer.show_loading_effect()
        if image2 is not None:
            self.hybrid_image_window.second_filtered_image_viewer.show_loading_effect()
        self.hybrid_image_window.controls_container.setEnabled(False)
        self.hybrid_image_window.image_viewers_container.setEnabled(False)

        process = mp.Process(target = hybrid_image_process,args = (image1,image2,radius,type1,type2,self.queue))
        process.start()
        self._start_queue_timer()


        


    
    def plotting_images(self, image1, image2, mix,inverse_mix):
        if image1 is not None:
            self.hybrid_image_window.first_filtered_image_viewer.display_and_set_image_matrix(image1)

        if image2 is not None:
            self.hybrid_image_window.second_filtered_image_viewer.display_and_set_image_matrix(image2)

        if mix is not None:
            self.hybrid_image_window.hybrid_image_viewer.display_and_set_image_matrix(inverse_mix)

        


    def _start_queue_timer(self):
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self._check_queue)
        self.queue_timer.start(100)

    def _check_queue(self):
        if self.queue and not self.queue.empty():
            self.queue_timer.stop()
            img1, img2, mix,inverse_mix = self.queue.get()
            self.hybrid_image_window.hybrid_image_viewer.hide_loading_effect()
            if img1 is not None:
                self.hybrid_image_window.first_filtered_image_viewer.hide_loading_effect()
            if img2 is not None:
                self.hybrid_image_window.second_filtered_image_viewer.hide_loading_effect()
            self.hybrid_image_window.controls_container.setEnabled(True)
            self.hybrid_image_window.image_viewers_container.setEnabled(True)
            self.plotting_images(img1, img2, mix,inverse_mix)
            self.hybrid_image_window.show_toast(text = "Hybrid Image is complete.")        