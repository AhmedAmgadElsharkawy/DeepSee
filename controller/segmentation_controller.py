import cv2
import numpy as np
from sklearn.cluster import KMeans

class SegmentationController():
    def __init__(self,segmentation_window = None):
        self.segmentation_window = segmentation_window
        if self.segmentation_window:
            self.segmentation_window.apply_button.clicked.connect(self.apply_segmentation)


    def apply_segmentation(self):
        image = self.segmentation_window.input_image_viewer.image_model.get_image_matrix()
        k_value = self.segmentation_window.k_means_k_value_spin_box.value()
        max_iterations = self.segmentation_window.k_means_max_iterations_spin_box.value()
        
        
        region_threshold = self.segmentation_window.region_growing_threshold_spin_box.value()
        
        if self.segmentation_window.segmentation_algorithm_custom_combo_box.current_text() == "k-means":
            output_image = self.k_means_segmentation(image, k_value, max_iterations)
        elif self.segmentation_window.segmentation_algorithm_custom_combo_box.current_text() == "Mean Shift":    
            
            pass
        elif self.segmentation_window.segmentation_algorithm_custom_combo_box.current_text() == "Agglomerative Segmentation":    
           
            pass
        else:
           
            pass             
        self.segmentation_window.output_image_viewer.display_and_set_image_matrix(output_image)
        print("segmentation Done")


    def k_means_segmentation(self, image, k_value, max_iterations):
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape((-1, 3)).astype(np.float32)
        num_pixels = pixels.shape[0]
        
        indices = np.random.choice(num_pixels, k_value, replace=False)
        centers = pixels[indices]
        
        for _ in range(max_iterations):
            # Calculate distances from each pixel to each center
            distances = np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2)
            
            # Assign each pixel to the nearest center
            labels = np.argmin(distances, axis=1)
            
            # Update centers
            new_centers = np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                                    for i in range(k_value)])            
            # Check for convergence (if centers do not change)
            if np.all(centers == new_centers):
                break
            
            center = new_centers
        
        # map each pixel to its corresponding color
        segmented_image = centers[labels].reshape(image.shape).astype(np.uint8)
       
        return segmented_image

    