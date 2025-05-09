from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

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
        self.face_analysis_type_custom_combo_box.currentIndexChanged.connect(self.on_face_analysis_type_change)
        
        self.lowe_ratio_spin_box = CustomSpinBox(label="Lowe's Ratio",range_start=0,range_end=1,initial_value=0.6,step_value=0.01,decimals=2,double_value=True)
        self.pca_confidence_level_spin_box =  CustomSpinBox(label="PCA Confidence Level",range_start=0,range_end=1,initial_value=0.95,step_value=0.01,decimals=2,double_value=True)

        self.inputs_container_layout.addWidget(self.face_analysis_type_custom_combo_box)
        self.inputs_container_layout.addWidget(self.lowe_ratio_spin_box)
        self.inputs_container_layout.addWidget(self.pca_confidence_level_spin_box)

        self.on_face_analysis_type_change()

        self.face_detection_controller = FaceDetectionController(self)
        self.face_recogntion_controller = FaceRecognitionController(self)

    
    def on_face_analysis_type_change(self):
        selected_face_analysis_type = self.face_analysis_type_custom_combo_box.current_text()

        match selected_face_analysis_type:
            case "Face Recognition":
                self.lowe_ratio_spin_box.setVisible(True)
                self.pca_confidence_level_spin_box.setVisible(True)
            case "Face Detection":
                self.lowe_ratio_spin_box.setVisible(False)
                self.pca_confidence_level_spin_box.setVisible(False)


