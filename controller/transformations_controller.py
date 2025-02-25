import numpy as np
import cv2
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
    
    def grayScale_cdf(self, image, min_range=0, max_range=256):
        histogram = self.grayScale_histogram(image, min_range, max_range)
        cdf = histogram.cumsum()
        cdf_normalized = cdf * float(histogram.max()) / cdf.max()
        return cdf_normalized
    
    def rgb_cdf(self, image, min_range=0, max_range=256):
        blue_histogram, green_histogram, red_histogram = self.rgb_histogram(image, min_range, max_range)
        
        blue_cdf = blue_histogram.cumsum()
        green_cdf = green_histogram.cumsum()
        red_cdf = red_histogram.cumsum()
        
        blue_cdf_normalized = blue_cdf * float(blue_histogram.max()) / blue_cdf.max()
        green_cdf_normalized = green_cdf * float(green_histogram.max()) / green_cdf.max()
        red_cdf_normalized = red_cdf * float(red_histogram.max()) / red_cdf.max()
        
        return blue_cdf_normalized, green_cdf_normalized, red_cdf_normalized
    
    def get_cdf(self, image, min_range=0, max_range=256):
        if len(image.shape) == 2:
            return self.grayScale_cdf(image, min_range, max_range)  
        elif len(image.shape) == 3:
            return self.rgb_cdf(image, min_range, max_range)
        
    def equalize_grayScale(self, image):
        histogram = self.grayScale_histogram(image)
        cdf = self.grayScale_cdf(image)
        
        # Apply lookup table (CDF) for equalization
        equalized_image = cdf[image]

        return equalized_image
        
    def equalize_rgb(self, image):
        blue_histogram, green_histogram, red_histogram = self.rgb_histogram(image)
        blue_cdf, green_cdf, red_cdf = self.rgb_cdf(image)
        
        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]
        
        masked_cdf_blue = np.ma.masked_equal(blue_cdf, 0)
        masked_cdf_blue = (masked_cdf_blue) * 255 / (masked_cdf_blue.max())
        final_cdf_blue = np.ma.filled(masked_cdf_blue, 0).astype("uint8")

        masked_cdf_green = np.ma.masked_equal(green_cdf, 0)
        masked_cdf_green = (masked_cdf_green) * 255 / (masked_cdf_green.max())
        final_cdf_green = np.ma.filled(masked_cdf_green, 0).astype("uint8")

        masked_cdf_red = np.ma.masked_equal(red_cdf, 0)
        masked_cdf_red = (masked_cdf_red) * 255 / (masked_cdf_red.max())
        final_cdf_red = np.ma.filled(masked_cdf_red, 0).astype("uint8")

        image_blue = final_cdf_blue[blue]
        image_green = final_cdf_green[green]
        image_red = final_cdf_red[red]

        equalized_image = cv2.merge((image_blue, image_green, image_red))
        return equalized_image
    def equalize_image(self, image):
        if len(image.shape) == 2:
            return self.equalize_grayScale(image)
        elif len(image.shape) == 3:
            return self.equalize_rgb(image)