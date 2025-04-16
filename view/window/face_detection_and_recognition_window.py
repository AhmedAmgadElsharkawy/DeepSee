from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox

from controller.face_detection_controller import FaceDetectionController
from controller.face_recognition_controller import FaceRecognitionController

class FaceDetectionAndRecognitionWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(FaceDetectionAndRecognitionWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if FaceDetectionAndRecognitionWindow.__instance != None:
            return
        
        super().__init__(main_window, "Face Detection & Recognition")
        FaceDetectionAndRecognitionWindow.__instance =self

        self.face_analysis_type_custom_combo_box = CustomComboBox(label= "Face Analysis Type",combo_box_items_list=["Face Detection","Face Recognition"])
        self.inputs_container_layout.addWidget(self.face_analysis_type_custom_combo_box)

        self.face_detection_controller = FaceDetectionController(self)
        self.face_recogntion_controller = FaceRecognitionController(self)

    


