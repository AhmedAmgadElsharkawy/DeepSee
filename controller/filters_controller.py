import numpy as np
import cv2
import utils.utils as utils

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
                _, filtered_img = self.apply_low_or_high_pass_filter(image, Radius=raduis,Type="low")

            elif filter_type == "High Pass Filter":
                raduis = self.filters_window.high_pass_filter_radius_spin_box.value()
                _, filtered_img = self.apply_low_or_high_pass_filter(image, Radius=raduis,Type="high")
            else:

                print("Incorrect filter type")
            self.filters_window.output_image_viewer.display_and_set_image_matrix(filtered_img)
        except Exception as e:
            print(f"Error: {e}")



    def average_filter(self, image, kernel_size=3):


        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
        output_image = utils.convolution(image, kernel)
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return output_image


    def gaussian_filter(self, image, kernel_size=3, sigma=1):

        # Gaussian kernel
        kernel = utils.gaussian_kernel(kernel_size, sigma)
        output_image = utils.convolution(image, kernel)
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return output_image

    def median_filter(self, image, kernel_size=3):
        # Pading
        pad_size = kernel_size // 2
        padded_image = utils.pad_image(image=image, pad_size=pad_size)

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



    def compute_ifft(self, fft_shifted):    # inverse fourier transform
        fft_shifted_back = np.fft.ifftshift(fft_shifted)
        image_filtered = np.fft.ifft2(fft_shifted_back)
        return np.abs(image_filtered)



    def create_mask(self, shape, radius,type):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]        # mesh grid
        if type == "low":
            mask = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2) <= radius
        elif type == "high":
            mask = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2) > radius
        return mask.astype(np.float32)


    def apply_low_or_high_pass_filter(self, image, Radius=2,Type="low"):
        dft_shift = self.compute_fft(image)
        mask = self.create_mask(image.shape, Radius,Type)
        dft_shift_filtered = dft_shift * mask
        return dft_shift_filtered, self.compute_ifft(dft_shift_filtered)

