import cv2
import numpy as np

from controller.transformations_controller import TransformationsController

class ImageModel:
    def __init__(self):
        self.image_matrix = None 
        self.gray_image_matrix = None

    def load_image(self, file_path):
        with open(file_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

        temp_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        self.image_matrix = temp_image
        if temp_image.ndim == 2:
            self.gray_image_matrix = temp_image
        else:
            self.gray_image_matrix = TransformationsController.grayscale_image(temp_image)

    def save_image(self, save_path):
        if self.image_matrix is not None:
            ext = save_path.split('.')[-1] 
            success, buffer = cv2.imencode(f'.{ext}', self.image_matrix)

            if success:
                with open(save_path, "wb") as f:
                    f.write(buffer)

    def get_image_matrix(self):
        return self.image_matrix
    
    def set_image_matrix(self,matrix):
        self.image_matrix = matrix.copy()
        if self.image_matrix.ndim == 3:
            self.gray_image_matrix = TransformationsController.grayscale_image(self.image_matrix)
        elif self.image_matrix.ndim == 2:
            self.gray_image_matrix=self.image_matrix


    def set_gray_image_matrix(self, matrix):
        self.gray_image_matrix = matrix.copy()

    def get_gray_image_matrix(self):
        return self.gray_image_matrix
    
    def is_grayscale(self,image_matrix):
        return len(image_matrix.shape) < 3

    def reset(self):
        self.image_matrix = None 
        self.gray_image_matrix = None
   
