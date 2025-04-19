import numpy as np
import multiprocessing as mp
from PyQt5.QtCore import QTimer



def add_gaussian_noise(image,mean,sigma,queue = None):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    if queue:
        queue.put(noisy_image)
    else:
        return noisy_image
    
def add_uniform_noise(image,noise_level,queue = None):
    noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    if queue:
        queue.put(noisy_image)
    else:
        return noisy_image
    
def add_salt_and_pepper_noise(image,salt,pepper,queue = None):
    noisy_image = image.copy()
    num_of_salt_pixels = int(salt * image.size)
    num_of_pepper_pixels = int(pepper * image.size) 

    coords = [np.random.randint(0, i - 1, num_of_salt_pixels) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i - 1, num_of_pepper_pixels) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    if queue:
        queue.put(noisy_image)
    else:
        return noisy_image


class NoiseController():
    def __init__(self,noise_window = None):
        self.noise_window = noise_window
        if self.noise_window:
            self.noise_window.apply_button.clicked.connect(self.apply_noise)


    def apply_noise(self):
        self.noise_window.output_image_viewer.show_loading_effect()
        self.noise_window.controls_container.setEnabled(False)
        self.noise_window.image_viewers_container.setEnabled(False)

        self.queue = mp.Queue()


        type = self.noise_window.noise_type_custom_combo_box.current_text()
        image = self.noise_window.input_image_viewer.image_model.get_image_matrix()
        if type == "Gaussian Noise":
            mean = self.noise_window.guassian_noise_mean_spin_box.value()
            sigma = self.noise_window.guassian_noise_sigma_spin_box.value()
            process = mp.Process(target = add_gaussian_noise,args=(image,mean,sigma,self.queue))
        elif type == "Uniform Noise":
            noise_level = self.noise_window.noise_value_spin_box.value()
            process = mp.Process(target = add_gaussian_noise,args=(image,noise_level,self.queue))
        else:
            salt = self.noise_window.salt_probability_spin_box.value() / 100
            pepper = self.noise_window.pepper_probability_spin_box.value() / 100
            process = mp.Process(target = add_salt_and_pepper_noise,args=(image,salt,pepper,self.queue))

        process.start()
        self._start_queue_timer()


    def _start_queue_timer(self):
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self._check_queue)
        self.queue_timer.start(100)

    def _check_queue(self):
        if self.queue and not self.queue.empty():
            self.queue_timer.stop()
            self.noise_window.output_image_viewer.hide_loading_effect()
            self.noise_window.controls_container.setEnabled(True)
            self.noise_window.image_viewers_container.setEnabled(True)
            result = self.queue.get()
            self.noise_window.output_image_viewer.display_and_set_image_matrix(result)


        

    