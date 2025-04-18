import numpy as np
import cv2
import pyqtgraph as pg

class TransformationsController():
    def __init__(self, transformations_window = None):
        self.transformations_window = transformations_window
        if self.transformations_window:
            self.transformations_window.apply_button.clicked.connect(self.apply_transformation)

    def apply_transformation(self):

        """Applies the selected transformation and updates the UI."""
        # Get the selected transformation type
        transformation_type = self.transformations_window.transformation_type_custom_combo_box.current_text()

        # Get input image from the UI
        input_image = self.transformations_window.input_image_viewer.image_model.get_image_matrix()
        if input_image is None:
            print("No image loaded")
            return

        transformed_image = None

        if transformation_type == "Grayscale" and  len(input_image.shape) == 2:
            transformed_image = input_image
        elif transformation_type == "Grayscale":
            transformed_image = self.grayscale_image(input_image)
        elif transformation_type == "Equalization":
            transformed_image = self.equalize_image(input_image)
        elif transformation_type == "Normalization":
            transformed_image = self.normalize_image(input_image)

        if transformed_image is not None:
            self.update_ui(input_image, transformed_image)



    def update_ui(self, original, transformed):
        """Updates the UI with the transformed image and histograms."""

        self.transformations_window.input_image_viewer.display_and_set_image_matrix((original))
        self.transformations_window.output_image_viewer.display_and_set_image_matrix((transformed))

        # Compute and update histograms and CDFs
        self.update_histograms(original, transformed)

    def update_histograms(self, original, transformed):
        # Compute histograms
        original_histogram = self.get_histogram(original)
        transformed_histogram = self.get_histogram(transformed)
        
        # Compute CDFs
        original_cdf = self.get_cdf(original)
        transformed_cdf = self.get_cdf(transformed)
        
        # Compute PDFs
        original_pdf = self.get_pdf(original)
        transformed_pdf = self.get_pdf(transformed)


        # Check if original is color (3 channels)
        if len(original_histogram) == 3:
            blue_hist_orig, green_hist_orig, red_hist_orig = original_histogram
            blue_cdf_orig, green_cdf_orig, red_cdf_orig = original_cdf
        elif len(original_histogram) == 256:  # Grayscale case
            gray_hist_orig = original_histogram
            gray_cdf_orig = original_cdf
            
        else:
            raise ValueError(f"Unexpected original histogram format: {len(original_histogram)}")

        # Check if transformed is grayscale or color
        if len(transformed_histogram) == 3:
            blue_hist_trans, green_hist_trans, red_hist_trans = transformed_histogram
            blue_cdf_trans, green_cdf_trans, red_cdf_trans = transformed_cdf
        elif len(transformed_histogram) == 256:  # Grayscale case
            gray_hist_trans = transformed_histogram
            gray_cdf_trans = transformed_cdf
       
        else:
            raise ValueError(f"Unexpected transformed histogram format: {len(transformed_histogram)}")

        

        # Clear previous plots
        self.transformations_window.orignal_image_histogram_graph.clear()
        self.transformations_window.transformed_image_histogram_graph.clear()
        self.transformations_window.orignal_image_cdf_graph.clear()
        self.transformations_window.transformed_image_cdf_graph.clear()
        self.transformations_window.orignal_image_pdf_graph.clear()
        self.transformations_window.transformed_image_pdf_graph.clear()

        # Semi-transparent colors using pg.mkBrush(R, G, B, Alpha)
        blue_brush = pg.mkBrush(0, 0, 255, 20)   # Blue with transparency
        green_brush = pg.mkBrush(0, 255, 0, 20)  # Green with transparency
        red_brush = pg.mkBrush(255, 0, 0, 20)    # Red with transparency
        gray_brush = pg.mkBrush(128, 128, 128, 20)  # Gray with transparency

        if len(original_histogram) == 3:  # Color image case
            # Convert to NumPy arrays
            blue_hist_orig, green_hist_orig, red_hist_orig = map(np.array, [blue_hist_orig, green_hist_orig, red_hist_orig])
            blue_cdf_orig, green_cdf_orig, red_cdf_orig = map(np.array, [blue_cdf_orig, green_cdf_orig, red_cdf_orig])
            
            # Plot original histograms
            self.transformations_window.orignal_image_histogram_graph.plot(red_hist_orig, pen="r", fillLevel = 0, brush = red_brush, name="Red (Original)")
            self.transformations_window.orignal_image_histogram_graph.plot(green_hist_orig, pen="g",fillLevel = 0, brush = green_brush, name="Green (Original)")
            self.transformations_window.orignal_image_histogram_graph.plot(blue_hist_orig, pen="b",fillLevel = 0, brush = blue_brush, name="Blue (Original)")

            # Plot original CDFs
            self.transformations_window.orignal_image_cdf_graph.plot(red_cdf_orig, pen="r", fillLevel = 0, brush = red_brush, name="Red CDF (Original)")
            self.transformations_window.orignal_image_cdf_graph.plot(green_cdf_orig, pen="g", fillLevel = 0, brush = green_brush,name="Green CDF (Original)")
            self.transformations_window.orignal_image_cdf_graph.plot(blue_cdf_orig, pen="b", fillLevel = 0, brush = blue_brush, name="Blue CDF (Original)")
        else:  # Grayscale case
            gray_hist_orig = np.array(gray_hist_orig)
            gray_cdf_orig = np.array(gray_cdf_orig)

            self.transformations_window.orignal_image_histogram_graph.plot(gray_hist_orig, pen="k", fillLevel = 0, brush = gray_brush, name="Grayscale (Transformed)")
            self.transformations_window.orignal_image_cdf_graph.plot(gray_cdf_orig, pen="k", fillLevel = 0, brush = gray_brush, name="Grayscale CDF (Transformed)")
        
        
        # Plot transformed histogram (color or grayscale)
        if len(transformed_histogram) == 3:  # Color image case
            blue_hist_trans, green_hist_trans, red_hist_trans = map(np.array, [blue_hist_trans, green_hist_trans, red_hist_trans])
            blue_cdf_trans, green_cdf_trans, red_cdf_trans = map(np.array, [blue_cdf_trans, green_cdf_trans, red_cdf_trans])

            self.transformations_window.transformed_image_histogram_graph.plot(blue_hist_trans, pen="b", fillLevel = 0, brush = blue_brush, name="Blue (Transformed)")
            self.transformations_window.transformed_image_histogram_graph.plot(green_hist_trans, pen="g", fillLevel = 0, brush = green_brush, name="Green (Transformed)")
            self.transformations_window.transformed_image_histogram_graph.plot(red_hist_trans, pen="r", fillLevel = 0, brush = red_brush, name="Red (Transformed)")

            # Plot transformed CDFs
            self.transformations_window.transformed_image_cdf_graph.plot(blue_cdf_trans, pen="b", fillLevel = 0, brush = blue_brush, name="Blue CDF (Transformed)")
            self.transformations_window.transformed_image_cdf_graph.plot(green_cdf_trans, pen="g", fillLevel = 0, brush = green_brush, name="Green CDF (Transformed)")
            self.transformations_window.transformed_image_cdf_graph.plot(red_cdf_trans, pen="r", fillLevel = 0, brush = red_brush, name="Red CDF (Transformed)")

        else:  # Grayscale case
            gray_hist_trans = np.array(gray_hist_trans)
            gray_cdf_trans = np.array(gray_cdf_trans)

            self.transformations_window.transformed_image_histogram_graph.plot(gray_hist_trans, pen="k", fillLevel = 0, brush = gray_brush, name="Grayscale (Transformed)")
            self.transformations_window.transformed_image_cdf_graph.plot(gray_cdf_trans, pen="k", fillLevel = 0, brush = gray_brush, name="Grayscale CDF (Transformed)")
        

    @staticmethod
    def grayscale_image(image):
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    def grayscale_histogram(self, image, min_range=0, max_range=256):
        histogram, _ = np.histogram(image.flatten(), bins=256, range=(min_range, max_range))
        return histogram
    
    def rgb_histogram(self, image, min_range=0, max_range=256):
        blue_histogram, _ = np.histogram(image[:, :, 0].flatten(), bins=256, range=(min_range, max_range))
        green_histogram, _ = np.histogram(image[:, :, 1].flatten(), bins=256, range=(min_range, max_range))
        red_histogram, _ = np.histogram(image[:, :, 2].flatten(), bins=256, range=(min_range, max_range))
        return blue_histogram, green_histogram, red_histogram

    # Get histogram of an image
    def get_histogram(self, image):
        if len(image.shape) == 2:  # Grayscale image
            return self.grayscale_histogram(image)
        elif len(image.shape) == 3:  # RGB image
            return self.rgb_histogram(image)
        return image
        # image = self.grayscale_image(image)
        # return self.grayscale_histogram(image)
    
    def grayscale_cdf(self, image):
        histogram = self.grayscale_histogram(image)
        cdf = histogram.cumsum()
        cdf_normalized = cdf / float(cdf.max())  # Normalize CDF
        return cdf_normalized
    
    def rgb_cdf(self, image):
        blue_histogram, green_histogram, red_histogram = self.rgb_histogram(image)

        blue_histogram = blue_histogram.astype(np.float64)
        green_histogram = green_histogram.astype(np.float64)
        red_histogram = red_histogram.astype(np.float64)

        blue_cdf = np.cumsum(blue_histogram)
        green_cdf = np.cumsum(green_histogram)
        red_cdf = np.cumsum(red_histogram)

        blue_cdf_normalized = blue_cdf / blue_cdf.max()
        green_cdf_normalized = green_cdf / green_cdf.max()
        red_cdf_normalized = red_cdf / red_cdf.max()

        return blue_cdf_normalized, green_cdf_normalized, red_cdf_normalized
    
    def get_cdf(self, image):
        if len(image.shape) == 2:
            return self.grayscale_cdf(image)  
        elif len(image.shape) == 3:
            return self.rgb_cdf(image)
        return image

    def grayscale_pdf(self, image):
        histogram = self.grayscale_histogram(image)
        total_pixels = image.size
        pdf = histogram / total_pixels  # Normalize to probability
        return pdf

    def rgb_pdf(self, image):
        """Computes the PDF for each RGB channel."""
        blue_histogram, green_histogram, red_histogram = self.rgb_histogram(image)
        total_pixels = image.shape[0] * image.shape[1]

        blue_pdf = blue_histogram / total_pixels
        green_pdf = green_histogram / total_pixels
        red_pdf = red_histogram / total_pixels

        return blue_pdf, green_pdf, red_pdf

    def get_pdf(self, image):
        """Returns the PDF of the given image."""
        if len(image.shape) == 2:  # Grayscale
            return self.grayscale_pdf(image)
        elif len(image.shape) == 3:  # RGB
            return self.rgb_pdf(image)

    def equalize_grayscale(self, image):
        histogram = self.grayscale_histogram(image)
        # Calculate cumulative distribution function (CDF)
        cdf = histogram.copy()
        cdf = np.cumsum(cdf)

        # Normalize CDF
        cdf = cdf / cdf[-1]
        cdf = np.round(cdf * 255).astype(np.uint8)  # Map to intensity range

        # Apply lookup table (CDF) for equalization
        equalized_image = cdf[image]

        return equalized_image

    def equalize_rgb(self, image):
        blue, green, red = cv2.split(image)

        histogram_blue, _ = np.histogram(blue.flatten(), 256, [0, 256])
        histogram_green, _ = np.histogram(green.flatten(), 256, [0, 256])
        histogram_red, _ = np.histogram(red.flatten(), 256, [0, 256])

        cdf_blue = np.cumsum(histogram_blue)
        cdf_green = np.cumsum(histogram_green)
        cdf_red = np.cumsum(histogram_red)

        masked_cdf_blue = np.ma.masked_equal(cdf_blue, 0)
        masked_cdf_blue = (masked_cdf_blue) * 255 / (masked_cdf_blue.max())
        final_cdf_blue = np.ma.filled(masked_cdf_blue, 0).astype("uint8")

        masked_cdf_green = np.ma.masked_equal(cdf_green, 0)
        masked_cdf_green = (masked_cdf_green) * 255 / (masked_cdf_green.max())
        final_cdf_green = np.ma.filled(masked_cdf_green, 0).astype("uint8")

        masked_cdf_red = np.ma.masked_equal(cdf_red, 0)
        masked_cdf_red = (masked_cdf_red) * 255 / (masked_cdf_red.max())
        final_cdf_red = np.ma.filled(masked_cdf_red, 0).astype("uint8")

        image_blue = final_cdf_blue[blue]
        image_green = final_cdf_green[green]
        image_red = final_cdf_red[red]

        equalized_image = cv2.merge((image_blue, image_green, image_red))
        return equalized_image
        
    # Get the equalized image
    def equalize_image(self, image):
        if len(image.shape) == 2:
            image = self.equalize_grayscale(image)
        elif len(image.shape) == 3:
            image = self.equalize_rgb(image)
        return image
    
    def normalize_grayscale(self, image):
        normalized_image = image.astype(np.float32)
        min_value = np.min(normalized_image)
        max_value = np.max(normalized_image)
        range_value = max_value - min_value
        
        if range_value == 0:  # Avoid division by zero
            return np.zeros_like(normalized_image)
        normalized_image = (normalized_image - min_value) / range_value
        return (normalized_image * 255).astype(np.uint8)

    def normalize_rgb(self, image):
        normalized_image = image.astype(np.float32)
        
        for i in range(3):  # Iterate over R, G, B channels
            min_value = np.min(normalized_image[:, :, i])
            max_value = np.max(normalized_image[:, :, i])
            range_value = max_value - min_value
            
            if range_value == 0:
                normalized_image[:, :, i] = 0  # Set to zero if no variation
            else:
                normalized_image[:, :, i] = (normalized_image[:, :, i] - min_value) / range_value

        return (normalized_image * 255).astype(np.uint8)


    def normalize_image(self, image):
        """Handles grayscale and RGB image normalization."""
        if len(image.shape) == 2:
            return self.normalize_grayscale(image)
        elif len(image.shape) == 3:
            return self.normalize_rgb(image)
        return image  # Return unchanged if format is unknown
    

