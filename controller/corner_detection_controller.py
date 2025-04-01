class CornerDetectionController():
    def __init__(self,corner_detection_window):
        self.corner_detection_window = corner_detection_window
        self.corner_detection_window.apply_button.clicked.connect(self.apply_corner_detection)

    def apply_corner_detection(self):
        image=self.corner_detection_window.input_image_viewer.image_model.get_gray_image_matrix()
        print(image.shape)
        print("corner detection")

