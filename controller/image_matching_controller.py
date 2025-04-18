import cv2
import numpy as np
from controller.sift_descriptors_controller import SiftDescriptorsController

class ImageMatchingController():
    def __init__(self,image_matching_window):
        self.image_matching_window = image_matching_window
        self.image_matching_window.apply_button.clicked.connect(self.apply_image_matching)
        self.sift_descriptors_controller = SiftDescriptorsController()


    def apply_image_matching(self):
        img_sigma = self.image_matching_window.img_detect_keypoints_sigma_spin_box.value()
        img_num_intervals = self.image_matching_window.img_detect_keypoints_intervals_number_spin_box.value()
        img_assumed_blur = self.image_matching_window.img_detect_keypoints_assumed_blur_spin_box.value()

        template_sigma = self.image_matching_window.template_detect_keypoints_sigma_spin_box.value()
        template_num_intervals = self.image_matching_window.template_detect_keypoints_intervals_number_spin_box.value()
        template_assumed_blur = self.image_matching_window.template_detect_keypoints_assumed_blur_spin_box.value()

        selected_matching_algorithm = self.image_matching_window.matching_algorithm_custom_combo_box.current_text()
        img = self.image_matching_window.input_image_viewer.image_model.gray_image_matrix
        tempelate = self.image_matching_window.input_template_viewer.image_model.gray_image_matrix
        gray_img = self.image_matching_window.input_image_viewer.image_model.gray_image_matrix.astype('float32')
        gray_template = self.image_matching_window.input_template_viewer.image_model.gray_image_matrix.astype('float32')

        if gray_img is None or gray_template is None:
            return

        kp1, desc1 = self.sift_descriptors_controller.get_sift_keypoints_and_descriptors(gray_img,img_sigma,img_assumed_blur,img_num_intervals)
        kp2, desc2 = self.sift_descriptors_controller.get_sift_keypoints_and_descriptors(gray_template,template_sigma,template_assumed_blur,template_num_intervals)

        matches = None

        if selected_matching_algorithm == "Sum Of Squared Differences":
            matches = self.match_ssd(desc1,desc2)
        elif selected_matching_algorithm == "Normalized Cross Correlation":
            matches = self.match_ncc(desc1,desc2)

        if matches is None:
            return
        
        img_with_matches = cv2.drawMatches(img, kp1, tempelate, kp2, matches[:100], None, flags=2)

        self.image_matching_window.output_image_viewer.display_and_set_image_matrix(img_with_matches)


    def match_ssd(self,desc1, desc2, ratio_threshold=0.4):
        matches = []
        for i, d1 in enumerate(desc1):
            ssd = np.sum((desc2 - d1) ** 2, axis=1)
            sorted_indices = np.argsort(ssd)  
            best_match_idx = sorted_indices[0]
            second_best_match_idx = sorted_indices[1]
            if ssd[best_match_idx] < ratio_threshold * ssd[second_best_match_idx]:
                matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_match_idx, _distance=ssd[best_match_idx]))
        return matches


    def normalize_desc(self,desc):
        return desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-10)
    

    def match_ncc(self,desc1, desc2, threshold=0.9):  
        desc1_n = self.normalize_desc(desc1)
        desc2_n = self.normalize_desc(desc2)
        matches = []
        for i, d1 in enumerate(desc1_n):
            sim = np.dot(desc2_n, d1)
            j = np.argmax(sim)
            if sim[j] > threshold:  
                matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=1 - sim[j]))
        return matches
        