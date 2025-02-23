import numpy as np
import cv2

class ImageModel:
    def __init__(self):
        self.image_matrix = None 

    def load_image(self, file_path):
        self.image_matrix = cv2.imread(file_path)
            

    def save_image(self, save_path):
        if self.image_matrix is not None:
            cv2.imwrite(save_path, self.image_matrix)

    def get_image_matrix(self):
        return self.image_matrix
    
    def set_image_matrix(self,matrix):
        self.image_matrix = matrix.copy()
        
    def correlate2d(self,matrix,kernel):
        pass 
   
