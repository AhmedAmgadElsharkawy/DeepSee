class ImageMatchingController():
    def __init__(self,image_matching_window):
        self.image_matching_window = image_matching_window
        self.image_matching_window.apply_button.clicked.connect(self.apply_image_matching)


    def apply_image_matching(self):
        print("Image Matching")


        