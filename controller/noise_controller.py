import numpy as np
import multiprocessing as mp
from PyQt5.QtCore import QThread, pyqtSignal



"""
    we use multiprocessing to run the algorithms on seperate core from the gui

    The creation of the process take some time which freeezes the gui if run on the same therad so 
    use multithreading to do the logic of creating and tracking the process on different thread

    Create IPC queue to share resources between cores
    IPC stands for Inter‑Process Communication. 
    It’s the set of mechanisms an operating system (and your programs) provide 
    to let separate processes—each with its own private memory space—exchange data and signals.
    Without IPC, one process couldn’t directly read or write another process’s memory.

    any function you pass to multiprocessing.Process must be defined at the top level of a module, not 
    inside a class or another function. That way, Python’s pickler (serialize and deserialize data between processes memory) can import it by name

    not all objects can be easily pickled. For example:
    Functions or classes defined inside another function: These can't be pickled because they don't have a top-level name, and pickling requires that objects can be imported by name.
    Local variables: Variables that are only local to a function or method might also pose issues with pickling.  

    underscore before a function or variable name in Python is a convention that 
    This is a private/internal function or variable. Please don’t use it outside this class/module unless you really know what you’re doing. 
"""


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


class NoiseProcessWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self,type,params):
        super().__init__()
        self.params = params
        self.type = type

    def run(self):  
        queue = mp.Queue()

        if self.type == "Gaussian Noise":
            process = mp.Process(target = add_gaussian_noise,args=(self.params['image'],self.params['mean'],self.params['sigma'],queue))
        elif self.type == "Uniform Noise":
            process = mp.Process(target = add_uniform_noise,args=(self.params['image'],self.params['noise_level'],queue))
        else:
            process = mp.Process(target = add_salt_and_pepper_noise,args=(self.params['image'],self.params['salt'],self.params['pepper'],queue))

        process.start()
        while True:
            if not queue.empty():
                result = queue.get()
                self.result_ready.emit(result)
                break
            self.msleep(50)
        process.join()


class NoiseController():
    def __init__(self,noise_window = None):
        self.noise_window = noise_window
        if self.noise_window:
            self.noise_window.apply_button.clicked.connect(self.apply_noise)


    def apply_noise(self):
        self.noise_window.output_image_viewer.show_loading_effect()
        self.noise_window.controls_container.setEnabled(False)
        self.noise_window.image_viewers_container.setEnabled(False)

        params = {}

        params['type'] = self.noise_window.noise_type_custom_combo_box.current_text()
        params['image'] = self.noise_window.input_image_viewer.image_model.get_image_matrix()
        if type == "Gaussian Noise":
            params['mean'] = self.noise_window.guassian_noise_mean_spin_box.value()
            params['sigma'] = self.noise_window.guassian_noise_sigma_spin_box.value()
        elif type == "Uniform Noise":
            params['noise_level'] = self.noise_window.noise_value_spin_box.value()
        else:
            params['salt'] = self.noise_window.salt_probability_spin_box.value() / 100
            params['pepper'] = self.noise_window.pepper_probability_spin_box.value() / 100

        self.worker = NoiseProcessWorker(type,params)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()


    def _on_result(self,result):
        self.noise_window.output_image_viewer.hide_loading_effect()
        self.noise_window.controls_container.setEnabled(True)
        self.noise_window.image_viewers_container.setEnabled(True)
        self.noise_window.output_image_viewer.display_and_set_image_matrix(result)
        self.noise_window.show_toast(text = "Noise is complete.")      
  



        

    