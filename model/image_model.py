import cv2
from controller.transformations_controller import TransformationsController
import numpy as np

class ImageModel:
    def __init__(self):
        self.image_matrix = None 
        self.gray_image_matrix = None

    def load_image(self, file_path):
        with open(file_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

        self.image_matrix = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        self.gray_image_matrix = TransformationsController.grayscale_image(self.image_matrix)
  

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
   
