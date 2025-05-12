import numpy as np
import cv2
import os
import multiprocessing as mp
from PyQt5.QtCore import QThread, pyqtSignal

dataset_path = "data/train_faces"

def face_recognition_process(test_img, lowe_ratio, pca_confidence_level, queue = None):
    faces = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            faces.append(img.flatten().reshape(-1, 1))
    
    faces = np.array(faces)
    X = faces.reshape((faces.shape[0], -1)).T

    mean_face = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_face

    L = X_centered.T @ X_centered
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_variance = np.sum(eigvals)
    variance_ratio = eigvals / total_variance
    cumulative_variance = np.cumsum(variance_ratio)

    k = np.searchsorted(cumulative_variance, pca_confidence_level) + 1

    eigenfaces = X_centered @ eigvecs
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    eigenfaces = eigenfaces[:, :k]

    projections = eigenfaces.T @ X_centered

    test_centered = handle_test_image(test_img,mean_face)

    test_proj = eigenfaces.T @ test_centered

    distances = np.linalg.norm(projections - test_proj, axis=0)

    sorted_indices = np.argsort(distances)
    best_match = sorted_indices[0]
    second_best_match = sorted_indices[1]

    best_distance = distances[best_match]
    second_best_distance = distances[second_best_match]

    matched_face = None

    if best_distance < lowe_ratio * second_best_distance:
        matched_face = X[:, best_match].reshape(64, 64)

    if queue:
        queue.put(matched_face)
    else:
        return matched_face

def handle_test_image(test_img,mean_face, target_size=(64, 64)):
    if test_img.dtype != np.float32 and test_img.max() > 1:
        test_img = test_img.astype(np.float32) / 255.0

    if test_img.shape != target_size:
        test_img = cv2.resize(test_img, target_size, interpolation=cv2.INTER_NEAREST)

    flattened_test_img = test_img.flatten().reshape(-1, 1)
    test_centered = flattened_test_img - mean_face
    return test_centered

class FaceRecognitionWorker(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        queue = mp.Queue()
        
        process = mp.Process(target=face_recognition_process, args=(self.params['test_img'], self.params["lowe_ratio"], self.params["pca_confidence_level"], queue))
        process.start()

        while True:
            if not queue.empty():
                result = queue.get()
                self.result_ready.emit(result)
                break
            self.msleep(50)

        process.join()

class FaceRecognitionController():
    def __init__(self,Face_detection_and_recognition_window = None):
        self.Face_detection_and_recognition_window = Face_detection_and_recognition_window
        if self.Face_detection_and_recognition_window:
            self.Face_detection_and_recognition_window.apply_button.clicked.connect(self.apply_face_recognition)

    def apply_face_recognition(self):
        if self.Face_detection_and_recognition_window.face_analysis_type_custom_combo_box.current_text() == "Face Recognition":
            test_img = self.Face_detection_and_recognition_window.input_image_viewer.image_model.get_gray_image_matrix()
            lowe_ratio = self.Face_detection_and_recognition_window.lowe_ratio_spin_box.value()
            pca_confidence_level = self.Face_detection_and_recognition_window.pca_confidence_level_spin_box.value()
            
            if test_img is None:
                return
 
            self.Face_detection_and_recognition_window.output_image_viewer.show_loading_effect()
            self.Face_detection_and_recognition_window.controls_container.setEnabled(False)
            self.Face_detection_and_recognition_window.image_viewers_container.setEnabled(False)

            params = {"test_img" : test_img, "lowe_ratio" : lowe_ratio,"pca_confidence_level" : pca_confidence_level}
            self.worker = FaceRecognitionWorker(params)
            self.worker.result_ready.connect(self._on_result)
            self.worker.start()

    def _on_result(self,result_image):
        self.Face_detection_and_recognition_window.output_image_viewer.hide_loading_effect()
        self.Face_detection_and_recognition_window.controls_container.setEnabled(True)
        self.Face_detection_and_recognition_window.image_viewers_container.setEnabled(True)

        if result_image is None:
            self.Face_detection_and_recognition_window.output_image_viewer.reset()
            self.Face_detection_and_recognition_window.show_toast(title = "Failed!", text = "No Confient Match Found.",type="ERROR")  
        else:
            self.Face_detection_and_recognition_window.output_image_viewer.display_and_set_image_matrix(result_image)
            self.Face_detection_and_recognition_window.show_toast(title = "Success!", text = "Face is recognized successfully.")        