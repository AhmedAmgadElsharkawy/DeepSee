class ActiveContoursController():
    def __init__(self,active_contours_window):
        self.active_contours_window = active_contours_window
        self.active_contours_window.apply_button.clicked.connect(self.apply_active_contour)


    def apply_active_contour(self):
        print("apply_active_contour")


