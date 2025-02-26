import cv2
import numpy as np
from scipy.ndimage import convolve
class EdgeDetectionController():
    def __init__(self,edge_detection_window):
        self.edge_detection_window = edge_detection_window
        self.edge_detection_window.apply_button.clicked.connect(self.apply_edge_detection)

    def apply_edge_detection(self):
        type = self.edge_detection_window.edge_detector_type_custom_combo_box.current_text()
        image = self.edge_detection_window.input_image_viewer.image_model.get_image_matrix()
        if type == "Sobel Detector":
            result = self.sobel(image)
        elif type == "Roberts Detector":
            print("ROB")
        elif type == "Prewitt Detector":
            result = self.prewitt_edge_detection(image)
        else:
            result = self.canny(image)
        self.edge_detection_window.output_image_viewer.display_and_set_image_matrix(result)

    def sobel(self, image):
        # Apply Sobel operator in X and Y directions
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X-direction
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y-direction

        # Compute the gradient magnitude
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)

        # Convert to 8-bit for visualization
        sobel_combined = np.uint8(sobel_combined)
        return sobel_combined
    
    def prewitt(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)  # Horizontal edges
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)  # Vertical edges

        # Apply convolution with Prewitt kernels
        edge_x = convolve(gray_image, prewitt_x)
        edge_y = convolve(gray_image, prewitt_y)

        # Compute gradient magnitude
        prewitt_combined = np.sqrt(edge_x**2 + edge_y**2)
        prewitt_combined = (prewitt_combined / np.max(prewitt_combined) * 255).astype(np.uint8) 
        return prewitt_combined
    
    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 10, 20)
        return edges
    
    def convolve2D(self, image, kernel):
        """
        Perform 2D convolution of an image with a kernel.
        
        Args:
            image (numpy.ndarray): Input image (2D array).
            kernel (numpy.ndarray): Convolution kernel (2D array).
        
        Returns:
            numpy.ndarray: Convolved image.
        """
        img_height, img_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Calculate padding size
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        # Pad the image
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        # Initialize output array
        output = np.zeros_like(image, dtype=np.float32)

        print(img_height, img_width)

        # Perform convolution
        for i in range(kernel_height):
            for j in range(kernel_width):
                output += kernel[i, j] * padded_image[i:i + img_height, j:j + img_width]

        return output

    def prewitt_edge_detection(self, image):
        """
        Perform Prewitt edge detection on an image.
        
        Args:
            image (numpy.ndarray): Input image (3D array, BGR format).
        
        Returns:
            numpy.ndarray: Edge-detected image (2D array).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define Prewitt kernels
        prewitt_x = np.array([[-1, 0, 1], 
                            [-1, 0, 1], 
                            [-1, 0, 1]], dtype=np.float32)

        prewitt_y = np.array([[-1, -1, -1], 
                            [0, 0, 0], 
                            [1, 1, 1]], dtype=np.float32)

        # Apply convolution
        grad_x = self.convolve2D(gray, prewitt_x)
        grad_y = self.convolve2D(gray, prewitt_y)

        # Compute gradient magnitude
        prewitt_edges = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255
        if prewitt_edges.max() > 0:
            prewitt_edges = (prewitt_edges / prewitt_edges.max() * 255).astype(np.uint8)
        else:
            prewitt_edges = np.zeros_like(prewitt_edges, dtype=np.uint8)

        return prewitt_edges