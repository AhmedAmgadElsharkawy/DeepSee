class HybridImageController():
    def __init__(self,hybrid_image_window):
        self.hybrid_image_window = hybrid_image_window
        self.hybrid_image_window.apply_button.clicked.connect(self.apply_image_mixing)

    def apply_image_mixing(self):
        print("applied")