import cv2
import multiprocessing as mp
from PyQt5.QtCore import QTimer

from controller.filters_controller import compute_ifft,apply_low_or_high_pass_filter
from PyQt5.QtCore import QThread, pyqtSignal



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


class HybridImageProcessWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self,params):
        super().__init__()
        self.params = params

    def run(self):
        queue = mp.Queue()
        process = mp.Process(target = hybrid_image_process,args = (self.params['image1'],self.params['image2'],self.params['radius'],self.params['type1'],self.params['type2'],queue))
        process.start()
        while True:
            if not queue.empty():
                result = {}
                result['img1'],result['img2'],result['mix'],result['inverse_mix'] = queue.get()
                self.result_ready.emit(result)
                break
            self.msleep(50)
        process.join()






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
        params = {}
        params['image1'] = self.hybrid_image_window.first_original_image_viewer.image_model.get_gray_image_matrix()
        params['image2'] = self.hybrid_image_window.second_original_image_viewer.image_model.get_gray_image_matrix()
        params['radius'] = self.hybrid_image_window.radius_custom_spin_box.value()
        params['type1'] = self.hybrid_image_window.first_image_filter_type_custom_combo_box.current_text()
        params['type2'] = self.hybrid_image_window.second_image_filter_type_custom_combo_box.current_text()
        

        self.hybrid_image_window.hybrid_image_viewer.show_loading_effect()
        if params['image1'] is not None:
            self.hybrid_image_window.first_filtered_image_viewer.show_loading_effect()
        if params['image2'] is not None:
            self.hybrid_image_window.second_filtered_image_viewer.show_loading_effect()
        self.hybrid_image_window.controls_container.setEnabled(False)
        self.hybrid_image_window.image_viewers_container.setEnabled(False)

        self.worker = HybridImageProcessWorker(params)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()


        
    def _on_result(self,result):
        self.hybrid_image_window.hybrid_image_viewer.hide_loading_effect()
        if result['img1'] is not None:
            self.hybrid_image_window.first_filtered_image_viewer.hide_loading_effect()
        if result['img2'] is not None:
            self.hybrid_image_window.second_filtered_image_viewer.hide_loading_effect()
        self.hybrid_image_window.controls_container.setEnabled(True)
        self.hybrid_image_window.image_viewers_container.setEnabled(True)
        self.plotting_images(result['img1'], result['img2'], result['mix'],result['inverse_mix'])
        self.hybrid_image_window.show_toast(text = "Hybrid Image is complete.")        

    
    def plotting_images(self, image1, image2, mix,inverse_mix):
        if image1 is not None:
            self.hybrid_image_window.first_filtered_image_viewer.display_and_set_image_matrix(image1)

        if image2 is not None:
            self.hybrid_image_window.second_filtered_image_viewer.display_and_set_image_matrix(image2)

        if mix is not None:
            self.hybrid_image_window.hybrid_image_viewer.display_and_set_image_matrix(inverse_mix)
            