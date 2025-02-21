class NoiseController():
    def __init__(self,noise_window):
        self.noise_window = noise_window
        self.noise_window.apply_button.clicked.connect(self.apply_noise)

    def apply_noise(self):
        print("applied")
    