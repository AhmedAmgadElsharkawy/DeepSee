class SegmentationController():
    def __init__(self,segmentation_window = None):
        self.segmentation_window = segmentation_window
        if self.segmentation_window:
            self.segmentation_window.apply_button.clicked.connect(self.apply_segmentation)


    def apply_segmentation(self):
        print("segmentation Done")


        