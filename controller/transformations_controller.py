import numpy as np

class TransformationsController():
    def __init__(self, transformations_window):
        self.transformations_window = transformations_window
        self.transformations_window.apply_button.clicked.connect(self.apply_transformation)

    def apply_transformation(self):
        print("applied")
        
    def grayScale_image(self, image):
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    def grayScale_histogram(self, image, min_range=0, max_range=256):
        histogram, _ = np.histogram(image.flatten(), bins=256, range=(min_range, max_range))
        return histogram
    
    def rgb_histogram(self, image, min_range=0, max_range=256):
        blue_histogram, _ = np.histogram(image[:, :, 0].flatten(), bins=256, range=(min_range, max_range))
        green_histogram, _ = np.histogram(image[:, :, 1].flatten(), bins=256, range=(min_range, max_range))
        red_histogram, _ = np.histogram(image[:, :, 2].flatten(), bins=256, range=(min_range, max_range))
        return blue_histogram, green_histogram, red_histogram

    # Get histogram of an image
    def get_histogram(self, image, min_range=0, max_range=256):
        if len(image.shape) == 2:  # Grayscale image
            return self.grayScale_histogram(image, min_range, max_range)
        elif len(image.shape) == 3:  # RGB image
            return self.rgb_histogram(image, min_range, max_range)
    
    def grayScale_cdf(self, image):
        histogram = self.grayScale_histogram(image)
        cdf = histogram.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize CDF
        return cdf_normalized.astype(np.uint8)
    
    def rgb_cdf(self, image):
        blue_histogram, green_histogram, red_histogram = self.rgb_histogram(image)
        
        blue_cdf = (blue_histogram.cumsum() - blue_histogram.min()) * 255 / (blue_histogram.max() - blue_histogram.min())
        green_cdf = (green_histogram.cumsum() - green_histogram.min()) * 255 / (green_histogram.max() - green_histogram.min())
        red_cdf = (red_histogram.cumsum() - red_histogram.min()) * 255 / (red_histogram.max() - red_histogram.min())

        return blue_cdf.astype(np.uint8), green_cdf.astype(np.uint8), red_cdf.astype(np.uint8)
    
    # Get CDF of an image
    def get_cdf(self, image, min_range=0, max_range=256):
            if len(image.shape) == 2:
                return self.grayScale_cdf(image, min_range, max_range)  
            elif len(image.shape) == 3:
                return self.rgb_cdf(image, min_range, max_range)
            
    def equalize_grayScale(self, image):
        cdf = self.grayScale_cdf(image)
        equalized_image = cdf[image]  # Map the original pixel values to equalized values
        return equalized_image

    def equalize_rgb(self, image):
        blue_cdf, green_cdf, red_cdf = self.rgb_cdf(image)

        image_blue = blue_cdf[image[:, :, 0]]
        image_green = green_cdf[image[:, :, 1]]
        image_red = red_cdf[image[:, :, 2]]

        equalized_image = np.stack((image_blue, image_green, image_red), axis=-1)
        return equalized_image.astype(np.uint8)
    
    # Get the equalized image
    def get_equalized_image(self, image):
        if len(image.shape) == 2:
            image = self.equalize_grayscale(image)
        elif len(image.shape) == 3:
            image = self.equalize_rgb(image)
        return image