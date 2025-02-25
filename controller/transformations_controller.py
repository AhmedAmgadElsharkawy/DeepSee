import numpy as np
class TransformationsController():
    def __init__(self,transformations_window):
        self.transformations_window = transformations_window
        self.transformations_window.apply_button.clicked.connect(self.apply_transformation)

    def apply_transformation(self):
        print("applied")
        
    def grayScale_image(self, image):
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    # compute grayscale histogram
    def grayScale_histogram(self, image, min_range = 0, max_range = 256):
        histogram , _ = np.histogram(image.flatten(), bins = 256, range = (min_range, max_range))
        return histogram
    
    # compute RGB histogram 
    def rgb_histogram(self, image, min_range = 0, max_range =256):
        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]
        
        blue_histogram , _ = np.histogram(blue.flatten(), bins = 256, range = (min_range, max_range))
        green_histogram , _ = np.histogram(green.flatten(), bins = 256, range = (min_range, max_range)) 
        red_histogram , _ = np.histogram(red.flatten(), bins = 256, range = (min_range, max_range)) 
        
        return blue_histogram, green_histogram, red_histogram
    
    # compute grayscale cumulative histogram
    def get_histogram(self, image, min_range=0, max_range=256):
        if len(image.shape) == 2:  # Grayscale image
            return self.grayScale_histogram(image, min_range, max_range)
        elif len(image.shape) == 3:  # RGB image
            return self.rgb_histogram(image, min_range, max_range)
    