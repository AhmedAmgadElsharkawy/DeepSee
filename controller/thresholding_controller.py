class ThresholdingController():
    def __init__(self,thresholding_window):
        self.thresholding_window = thresholding_window
        self.thresholding_window.apply_button.clicked.connect(self.apply_thresholding)

    def apply_thresholding(self):
        print("applied")