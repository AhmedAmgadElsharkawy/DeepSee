import numpy as np
import cv2
from controller.transformations_controller import TransformationsController

class ImageModel:
    def __init__(self):
        self.image_matrix = None 
        self.gray_image_matrix = None

    def load_image(self, file_path):
        self.image_matrix = cv2.imread(file_path)
        # self.image_matrix = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        self.gray_image_matrix = TransformationsController.grayscale_image(self.image_matrix)
            

    def save_image(self, save_path):
        if self.image_matrix is not None:
            cv2.imwrite(save_path, self.image_matrix)

    def get_image_matrix(self):
        return self.image_matrix
    
    def set_image_matrix(self,matrix):
        self.image_matrix = matrix.copy()

    def set_gray_image_matrix(self, matrix):
        self.gray_image_matrix = matrix.copy()

    def get_gray_image_matrix(self):
        return self.gray_image_matrix

    def reset(self):
        self.image_matrix = None 
        self.gray_image_matrix = None
   
