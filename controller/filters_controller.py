import numpy as np
import cv2

class FiltersController():
    def __init__(self,filters_window):
        self.filters_window = filters_window
        self.filters_window.apply_button.clicked.connect(self.apply_filter)

    def apply_filter(self):
        try:
            filter_type=self.filters_window.filter_type_custom_combo_box.current_text()
            image=self.filters_window.input_image_viewer.image_model.get_gray_image_matrix()
            print("Image type:", type(image))


            if filter_type == "Average Filter":
                kernel = self.filters_window.average_filter_kernel_size_spin_box.value()
                filtered_img = self.average_filter(image, kernel_size=kernel)

            elif filter_type == "Gaussian Filter":
                kernel = self.filters_window.guassian_filter_kernel_size_spin_box.value()
                variance=self.filters_window.guassian_filter_variance_spin_box.value()
                filtered_img = self.gaussian_filter(image, kernel_size=kernel, sigma=variance)

            elif filter_type == "Median Filter":
                kernel = self.filters_window.average_filter_kernel_size_spin_box.value()
                filtered_img = self.median_filter(image, kernel_size=kernel)

            elif filter_type == "Low Pass Filter":
                raduis = self.filters_window.low_pass_filter_radius_spin_box.value()
                filtered_img = self.apply_low_pass_filter(image, Raduis=raduis)

            elif filter_type == "High Pass Filter":
                raduis = self.filters_window.high_pass_filter_radius_spin_box.value()
                filtered_img = self.apply_high_pass_filter(image, Radius=raduis)
            else:

                print("Incorrect filter type")
            self.filters_window.output_image_viewer.display_image_matrix2(filtered_img)
        except Exception as e:
            print(f"Error: {e}")

    def pad_image(self, image, pad_size, pad_value=0):

        if image.ndim == 2:
            padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=pad_value)
        elif image.ndim == 3:
            padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=pad_value)
        else:
            raise ValueError("Unsupported image shape")

        return padded_image

    def average_filter(self, image, kernel_size=3):

        # Pading
        pad_size = kernel_size // 2
        padded_image = self.pad_image(image, pad_size)


        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)

        # Initialize the output image
        output_image = np.zeros_like(image, dtype=np.float32)

        # convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output_image[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)


        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return output_image
    def gaussian_kernel(self, size=3, sigma=1):
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        sum = 0

        # Gaussian values
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * 1 / (2 * np.pi * sigma**2)
                sum += kernel[i, j]

        return kernel / sum

    def gaussian_filter(self, image, kernel_size=3, sigma=1):

        # Pading
        pad_size = kernel_size // 2
        padded_image = self.pad_image(image, pad_size)

        # Gaussian kernel
        kernel = self.gaussian_kernel(kernel_size, sigma)

        # Initialize the output image
        output_image = np.zeros_like(image, dtype=np.float32)

        # convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output_image[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)

        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return output_image

    def median_filter(self, image, kernel_size=3):
        # Pading
        pad_size = kernel_size // 2
        padded_image = self.pad_image(image, pad_size)

        # Initialize the output image
        output_image = np.zeros_like(image, dtype=np.float32)


        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output_image[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])


        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return output_image

    def compute_fft(self, image): # fourier transform
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        return fft_shifted

    def create_lowpass_mask(self, shape, radius):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]     # mesh grid
        mask = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2) <= radius
        return mask.astype(np.float32)

    def create_highpass_mask(self, shape, radius):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]        # mesh grid
        mask = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2) > radius
        return mask.astype(np.float32)

    def compute_ifft(self, fft_shifted):    # inverse fourier transform
        fft_shifted_back = np.fft.ifftshift(fft_shifted)
        image_filtered = np.fft.ifft2(fft_shifted_back)
        return np.abs(image_filtered)

    def apply_low_pass_filter(self, image, Raduis=2):
        dft_shift = self.compute_fft(image)
        mask = self.create_lowpass_mask(image.shape, Raduis)
        dft_shift_filtered = dft_shift * mask
        return self.compute_ifft(dft_shift_filtered)

    def apply_high_pass_filter(self, image, Radius=2):
        dft_shift = self.compute_fft(image)
        mask = self.create_highpass_mask(image.shape, Radius)
        dft_shift_filtered = dft_shift * mask
        return self.compute_ifft(dft_shift_filtered)