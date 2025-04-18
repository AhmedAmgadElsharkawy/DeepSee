import cv2
import numpy as np
class CornerDetectionController():
    def __init__(self,corner_detection_window = None):
        self.corner_detection_window = corner_detection_window
        if self.corner_detection_window:
            self.corner_detection_window.apply_button.clicked.connect(self.apply_corner_detection)

    def apply_corner_detection(self):
        image=self.corner_detection_window.input_image_viewer.image_model.get_image_matrix()
        detected_corners_type=self.corner_detection_window.corner_detector_type_custom_combo_box.current_text()
        output_image=self.corner_detection_window.input_image_viewer.image_model.get_image_matrix()
        
        # Get the parameters from the UI
        kernel_size=self.corner_detection_window.harris_detector_kernel_size_spin_box.value()
        block_size=self.corner_detection_window.harris_detector_block_size_spin_box.value() 
        k=self.corner_detection_window.harris_detector_k_factor_spin_box.value()
        threshold=self.corner_detection_window.harris_detector_threshold_spin_box.value()
        max_corners=self.corner_detection_window.lambda_detector_max_corners_spin_box.value()
        min_distance=self.corner_detection_window.lambda_detector_min_distance_spin_box.value()
        quality_level=self.corner_detection_window.lambda_detector_quality_level_spin_box.value()

        if detected_corners_type == "Harris Detector":
            output_image = self.harris_corner_detector(image,block_size=block_size,ksize=kernel_size,k=k,threshold=threshold)
        elif detected_corners_type == "Lambda Detector":
            output_image = self.lambda_corner_detector(image,max_corners=max_corners,min_distance=min_distance,quality_level=quality_level)
        else:
            output_image = self.harris_and_lambda(image,block_size=block_size,ksize=kernel_size,k=k,threshold=threshold,
                                                  max_corners=max_corners,min_distance=min_distance,quality_level=quality_level)
            

        if output_image is not None:
            self.corner_detection_window.output_image_viewer.display_and_set_image_matrix(output_image)      


    def gray_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def draw_corners(self, image, corners: list) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        corners = np.array(corners).reshape(-1, 2)

        for corner in corners:
            cv2.circle(image, tuple(np.int32(corner[::-1])), 3, (0, 0, 255), -1)

        return image

    
    def harris_corner_detector(self,image,block_size:int = 2,ksize:int = 3,k:float = 0.04,threshold:float = 0.01)->list:
        
         # Convert image to grayscale if it is colored (3-channel)
        gray = self.gray_image(image)
            
        
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        if block_size % 2 == 0:
            block_size +=1
        Sxx = cv2.GaussianBlur(Ixx, (block_size, block_size), 1)
        Syy = cv2.GaussianBlur(Iyy, (block_size, block_size), 1)
        Sxy = cv2.GaussianBlur(Ixy, (block_size, block_size), 1)
        
        det_M = (Sxx * Syy) - (Sxy ** 2)
        trace_M = Sxx + Syy
        R = det_M - k * (trace_M ** 2)
        # Thresholding to get the corners
        corners = np.argwhere(R > threshold * R.max())
        return self.draw_corners(image, corners)
    
    def lambda_corner_detector(self, image, max_corners=10, min_distance=5, quality_level=0.01) -> list:
        # Convert image to grayscale
        gray = self.gray_image(image)

        # Compute image gradients
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Sum over window
        window_size = 5
    
        Sx2 = cv2.GaussianBlur(Ixx, (window_size, window_size), 1.5)
        Sy2 = cv2.GaussianBlur(Iyy, (window_size, window_size), 1.5)
        Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 1.5)

        # Compute minimum eigenvalue (lambda minus) response map
        h, w = gray.shape
        lambda_min = np.zeros((h, w), dtype=np.float32)
                              
        for y in range(h):
                    for x in range(w):
                        M = np.array([[Sx2[y, x], Sxy[y, x]],
                                    [Sxy[y, x], Sy2[y, x]]])
                        eigvals = np.linalg.eigvalsh(M)  # sorted: [lambda_min, lambda_max]
                        lambda_min[y, x] = eigvals[0]
       

        # Thresholding and NMS
        threshold = quality_level * lambda_min.max()
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(lambda_min, kernel)
        corners = np.column_stack(np.where((lambda_min == dilated) & (lambda_min > threshold)))

        # Sort by strength and apply min_distance
        corners = sorted(corners, key=lambda pt: -lambda_min[pt[0], pt[1]])
        final_corners = []
        for pt in corners:
            if all(np.linalg.norm(pt - fc) >= min_distance for fc in final_corners):
                final_corners.append(pt)
                if len(final_corners) >= max_corners:
                    break

        return self.draw_corners(image, final_corners)

              
    def harris_and_lambda(self,image,block_size:int = 2,ksize:int = 3,k:float = 0.04,threshold:float = 0.01
                          ,max_corners:int = 10,min_distance:int = 5,quality_level:float = 0.01)->list:
        # Convert image to grayscale if it is colored (3-channel)
        gray = self.gray_image(image)


        # Harris corner detection
        harris_corners = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
        harris_corners = cv2.dilate(harris_corners, None)
        # Thresholding to get the corners
        harris_corners_list = np.argwhere(harris_corners > threshold * harris_corners.max())

        # Lambda corner detection
        lambda_corners = cv2.cornerMinEigenVal(gray, blockSize=block_size, ksize=ksize)
        lambda_corners = np.int0(lambda_corners)
        
        harris_corners_list = np.array(harris_corners_list).reshape(-1, 2)
        lambda_corners = np.array(lambda_corners).reshape(-1, 2)

        all_corners = np.concatenate((harris_corners_list, lambda_corners), axis=0)
        return self.draw_corners(image, all_corners)

