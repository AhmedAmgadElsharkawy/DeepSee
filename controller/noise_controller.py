import numpy as np
class NoiseController():
    def __init__(self,noise_window):
        self.noise_window = noise_window
        self.noise_window.apply_button.clicked.connect(self.apply_noise)

    def apply_noise(self):
        type = self.noise_window.noise_type_custom_combo_box.current_text()
        image = self.noise_window.input_image_viewer.image_model.get_image_matrix()
        if type == "Gaussian Noise":
            result = self.add_gaussian_noise(image)
        elif type == "Uniform Noise":
            result = self.add_uniform_noise(image)
        else:
            result = self.add_salt_and_pepper_noise(image)
        self.noise_window.output_image_viewer.display_and_set_image_matrix(result)

    def add_gaussian_noise(self, image):
        mean = self.noise_window.guassian_noise_mean_spin_box.value()
        sigma = self.noise_window.guassian_noise_variance_spin_box.value()
        noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)
    
    def add_uniform_noise(self, image):
        noise_level = self.noise_window.noise_value_spin_box.value()
        noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)
    
    def add_salt_and_pepper_noise(self, image):
        noisy_image = image.copy()
        salt = self.noise_window.salt_probability_spin_box.value() / 100
        pepper = self.noise_window.pepper_probability_spin_box.value() / 100
        num_of_salt_pixels = int(salt * image.size)
        num_of_pepper_pixels = int(pepper * image.size) 

        coords = [np.random.randint(0, i - 1, num_of_salt_pixels) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 255

        coords = [np.random.randint(0, i - 1, num_of_pepper_pixels) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 0

        return noisy_image