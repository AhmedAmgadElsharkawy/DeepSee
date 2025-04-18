from PyQt5.QtWidgets import QWidget,QHBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_spin_box import CustomSpinBox

from controller.sift_descriptors_controller import SiftDescriptorsController

class SiftDescriptorsWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(SiftDescriptorsWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if SiftDescriptorsWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="SIFT Descriptors")
        SiftDescriptorsWindow.__instance = self


        self.detect_keypoints_inputs_container = QWidget()
        self.detect_keypoints_inputs_container_layout = QHBoxLayout(self.detect_keypoints_inputs_container)
        self.detect_keypoints_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.detect_keypoints_inputs_container)
        self.detect_keypoints_sigma_spin_box = CustomSpinBox(label="Sigma",range_start=0.1,range_end=100,initial_value=1.6,step_value=0.01,decimals=2,double_value=True)
        self.detect_keypoints_assumed_blur_spin_box = CustomSpinBox(label="Assumed Blur",range_start=0.01,range_end=100,initial_value=0.5,step_value=0.01,decimals=2,double_value=True)
        self.detect_keypoints_intervals_number_spin_box = CustomSpinBox(label="Intervals Number",range_start=1,range_end=5,initial_value=3,step_value=1)
        self.detect_keypoints_inputs_container_layout.addWidget(self.detect_keypoints_sigma_spin_box)
        self.detect_keypoints_inputs_container_layout.addWidget(self.detect_keypoints_assumed_blur_spin_box)
        self.detect_keypoints_inputs_container_layout.addWidget(self.detect_keypoints_intervals_number_spin_box)
        

        self.sift_controller = SiftDescriptorsController(self)
        
        

    


