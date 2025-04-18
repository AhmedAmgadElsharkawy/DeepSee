class FaceDetectionController():
    def __init__(self,Face_detection_and_recognition_window):
        self.Face_detection_and_recognition_window = Face_detection_and_recognition_window
        self.Face_detection_and_recognition_window.apply_button.clicked.connect(self.apply_face_detection)


    def apply_face_detection(self):
        if self.Face_detection_and_recognition_window.face_analysis_type_custom_combo_box.current_text() == "Face Detection":
            print("Face Detection")
        else:
            pass


        