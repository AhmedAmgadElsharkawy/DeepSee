import numpy as np
import cv2
import os

class FaceRecognitionController():
    def __init__(self,Face_detection_and_recognition_window = None):
        self.Face_detection_and_recognition_window = Face_detection_and_recognition_window
        if self.Face_detection_and_recognition_window:
            self.Face_detection_and_recognition_window.apply_button.clicked.connect(self.apply_face_recognition)


    def apply_face_recognition(self):
        if self.Face_detection_and_recognition_window.face_analysis_type_custom_combo_box.current_text() == "Face Recognition":
            faces = np.load("data/eigen_faces_dataset/olivetti_faces.npy")  
            X = faces.reshape((400, -1)).T 

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

            desired_confidence = 0.95  
            k = np.searchsorted(cumulative_variance, desired_confidence) + 1

            eigenfaces = X_centered @ eigvecs
            eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

            eigenfaces = eigenfaces[:, :k]

            projections = eigenfaces.T @ X_centered
                
            def handle_test_image(test_img, target_size=(64, 64)):
                if test_img.dtype != np.float32 and test_img.max() > 1:
                    test_img = test_img.astype(np.float32) / 255.0

                # Only resize if different shape
                if test_img.shape != target_size:
                    test_img = cv2.resize(test_img, target_size, interpolation=cv2.INTER_NEAREST)

                flattened_test_img = test_img.flatten().reshape(-1, 1)
                test_centered = flattened_test_img - mean_face
                return test_centered


            test_img = self.Face_detection_and_recognition_window.input_image_viewer.image_model.get_gray_image_matrix()

            test_centered = handle_test_image(test_img)

            # X_face_0_centered = X[:, 0].reshape(-1, 1) - mean_face
            # print("Max difference from faces[0]:", np.abs(test_centered - X_face_0_centered).max())

            test_proj = eigenfaces.T @ test_centered

            distances = np.linalg.norm(projections - test_proj, axis=0)

            sorted_indices = np.argsort(distances)
            best_match = sorted_indices[0]
            second_best_match = sorted_indices[1]

            best_distance = distances[best_match]
            second_best_distance = distances[second_best_match]

            ratio_threshold = 0.6  

            if best_distance < ratio_threshold * second_best_distance:
                matched_face = X[:, best_match].reshape(64, 64)
                self.Face_detection_and_recognition_window.output_image_viewer.display_and_set_image_matrix(matched_face)
            else:
                print("No confident match found.")
                self.Face_detection_and_recognition_window.output_image_viewer.reset()


        else:
            pass





        