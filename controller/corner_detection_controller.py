class CornerDetectionController():
    def __init__(self,corner_detection_window):
        self.corner_detection_window = corner_detection_window
        self.corner_detection_window.apply_button.clicked.connect(self.apply_corner_detection)

    def apply_corner_detection(self):
        print("corner detection")

