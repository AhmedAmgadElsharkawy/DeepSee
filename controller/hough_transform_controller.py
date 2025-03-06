class HoughTransformController():
    def __init__(self,hough_transform_window):
        self.hough_transform_window = hough_transform_window
        self.hough_transform_window.apply_button.clicked.connect(self.apply_hough_transform)


    def apply_hough_transform(self):
        detected_objects_type = self.hough_transform_window.detected_objects_type_custom_combo_box.current_text()

        match detected_objects_type:
            case "Lines Detection":
                self.detect_lines()
            case "Circles Detection":
                self.detect_circles()
            case "Ellipses Detection":
                self.detect_ellipses()


    def detect_lines(self):
        print("detect lines")

    def detect_circles(self):
        print("detect circles")

    def detect_ellipses(self):
        print("detect ellipses")
