class HoughTransformController():
    def __init__(self,hough_transform_window):
        self.hough_transform_window = hough_transform_window
        self.hough_transform_window.apply_button.clicked.connect(self.apply_hough_transform)


    def apply_hough_transform(self):
        detected_objects_type = self.hough_transform_window.detected_objects_type_custom_combo_box.current_text()
        input_image_matrix = self.hough_transform_window.input_image_viewer.image_model.get_image_matrix()
        output_image_matrix = None

        match detected_objects_type:
            case "Lines Detection":
                output_image_matrix = self.detect_lines(input_image_matrix)
            case "Circles Detection":
                output_image_matrix = self.detect_circles(input_image_matrix)
            case "Ellipses Detection":
                output_image_matrix = self.detect_ellipses(input_image_matrix)
            

        if output_image_matrix != None:
            self.hough_transform_window.output_image_viewer.display_and_set_image_matrix(output_image_matrix)


    def detect_lines(self,input_image_matrix):
        print("detect Lines")

    def detect_circles(self,input_image_matrix):
        print("detect circles")

    def detect_ellipses(self,input_image_matrix):
        print("detect ellipses")
