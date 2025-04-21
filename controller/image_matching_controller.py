import cv2
from multiprocessing import Process, Queue
import numpy as np
import time
from PyQt5.QtCore import QTimer

from controller.sift_descriptors_controller import get_sift_keypoints_and_descriptors


"""
    any function you pass to multiprocessing.Process must be defined at the top level of a module, not 
    inside a class or another function. That way, Python’s pickler (serialize and deserialize data between processes memory) can import it by name

    not all objects can be easily pickled. For example:
    Functions or classes defined inside another function: These can't be pickled because they don't have a top-level name, and pickling requires that objects can be imported by name.
    Local variables: Variables that are only local to a function or method might also pose issues with pickling.  


    underscore before a function or variable name in Python is a convention that 
    This is a private/internal function or variable. Please don’t use it outside this class/module unless you really know what you’re doing. 
"""


def process_image_matching(img_sigma, img_num_intervals, img_assumed_blur,img2_sigma, img2_num_intervals, img2_assumed_blur, selected_matching_algorithm, gray_img, gray_img2, img, img2,ssd_lowe_ratio,ncc_threshold, queue = None):
    start_time = time.time()

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray_img, None)
    kp2, desc2 = sift.detectAndCompute(gray_img2, None)


    # kp1, desc1 = get_sift_keypoints_and_descriptors(gray_img.astype('float32'), img_sigma, img_assumed_blur, img_num_intervals)
    # kp2, desc2 = get_sift_keypoints_and_descriptors(gray_img2.astype('float32'), img2_sigma, img2_assumed_blur, img2_num_intervals)

    matches = None
    if selected_matching_algorithm == "Sum Of Squared Differences":
        matches = match_ssd(desc1, desc2,ssd_lowe_ratio)
    elif selected_matching_algorithm == "Normalized Cross Correlation":
        matches = match_ncc(desc1, desc2,ncc_threshold)

    img_with_matches = cv2.drawMatches(img, kp1, img2, kp2, matches[:100], None, flags=2)

    elapsed_time = time.time() - start_time
    if queue:
        queue.put((elapsed_time, img_with_matches))
    else:
        return elapsed_time, img_with_matches

def match_ssd(desc1, desc2, lowe_ratio =0.2):
    matches = []
    for i, d1 in enumerate(desc1):
        ssd = np.sum((desc2 - d1) ** 2, axis=1)
        best, second = np.argsort(ssd)[:2]
        if ssd[best] < lowe_ratio * ssd[second]:
            matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best, _distance=ssd[best]))
    return matches

def normalize_desc(desc):
    return desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-10)

def match_ncc(desc1, desc2, threshold=0.97):
    desc1_n = normalize_desc(desc1)
    desc2_n = normalize_desc(desc2)
    matches = []
    for i, d1 in enumerate(desc1_n):
        sim = np.dot(desc2_n, d1)
        j = np.argmax(sim)
        if sim[j] > threshold:
            matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=1 - sim[j]))
    return matches



class ImageMatchingController:
    def __init__(self, image_matching_window=None):
        self.image_matching_window = image_matching_window
        self.process = None
        self.queue = None
        if self.image_matching_window:
            self.image_matching_window.apply_button.clicked.connect(self.apply_image_matching)

    def apply_image_matching(self):


        """
            Create IPC queue to share resources between cores
            IPC stands for Inter‑Process Communication. 
            It’s the set of mechanisms an operating system (and your programs) provide 
            to let separate processes—each with its own private memory space—exchange data and signals.
            Without IPC, one process couldn’t directly read or write another process’s memory.
        """


        img1 =  self.image_matching_window.input_image_viewer.image_model.image_matrix
        img2 =  self.image_matching_window.input_img2_viewer.image_model.image_matrix

        if(img1 is None or img2 is None):
            self.image_matching_window.show_toast(title = "Warning!", text = "Image Matching Invalid Images.",type="ERROR")      
            return  
        
        self.image_matching_window.output_image_viewer.show_loading_effect()
        self.image_matching_window.controls_container.setEnabled(False)
        self.image_matching_window.image_viewers_container.setEnabled(False)

        self.queue = Queue()


        args = (
                self.image_matching_window.img_detect_keypoints_sigma_spin_box.value(),
                self.image_matching_window.img_detect_keypoints_intervals_number_spin_box.value(),
                self.image_matching_window.img_detect_keypoints_assumed_blur_spin_box.value(),
                self.image_matching_window.img2_detect_keypoints_sigma_spin_box.value(),
                self.image_matching_window.img2_detect_keypoints_intervals_number_spin_box.value(),
                self.image_matching_window.img2_detect_keypoints_assumed_blur_spin_box.value(),
                self.image_matching_window.matching_algorithm_custom_combo_box.current_text(),
                self.image_matching_window.input_image_viewer.image_model.gray_image_matrix,
                self.image_matching_window.input_img2_viewer.image_model.gray_image_matrix,
                img1,
                img2,
                self.image_matching_window.ssd_lowe_ratio.value(),
                self.image_matching_window.ncc_threshold.value(),
            )

        self.process = Process(target=process_image_matching, args=(*args, self.queue))
        self.process.start()

        self._start_queue_timer()


    def _start_queue_timer(self):
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self._check_queue)
        self.queue_timer.start(100)

    def _check_queue(self):
        if self.queue and not self.queue.empty():
            elapsed_time, img_with_matches = self.queue.get()
            self.queue_timer.stop()
            self.image_matching_window.output_image_viewer.hide_loading_effect()
            self.image_matching_window.controls_container.setEnabled(True)
            self.image_matching_window.image_viewers_container.setEnabled(True)
            self.image_matching_window.time_elapsed_value.setText(f"{elapsed_time:.2f} Seconds")
            self.image_matching_window.output_image_viewer.display_and_set_image_matrix(img_with_matches)
            self.image_matching_window.show_toast(text = "Image Matching is complete.")        

