from PyQt5.QtWidgets import QWidget,QHBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

from controller.corner_detection_controller import CornerDetectionController

class CornerDetectionWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(CornerDetectionWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if CornerDetectionWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="Corner Detection")
        CornerDetectionWindow.__instance = self

        self.corner_detector_type_custom_combo_box = CustomComboBox(label= "Corner Detector Type",combo_box_items_list=["Harris Detector","Lambda Detector","Harris & Lambda Detector"])
        self.corner_detector_type_custom_combo_box.currentIndexChanged.connect(self.on_corner_detector_type_change)
        self.inputs_container_layout.addWidget(self.corner_detector_type_custom_combo_box)


        self.harris_detector_inputs_container = QWidget()
        self.harris_detector_inputs_container_layout = QHBoxLayout(self.harris_detector_inputs_container)
        self.harris_detector_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.harris_detector_inputs_container)
        self.harris_detector_block_size_spin_box = CustomSpinBox(label="Block Size",range_start=1,range_end=10,initial_value=2,step_value=1)
        self.harris_detector_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=11,initial_value=3,step_value=2)
        self.harris_detector_k_factor_spin_box = CustomSpinBox(label="K Factor",range_start=0,range_end=1000,initial_value=0.04,step_value=0.1,double_value=True,decimals=2)
        self.harris_detector_threshold_spin_box = CustomSpinBox(label="Threshold",range_start=0,range_end=1000,initial_value=0.01,step_value=0.1,decimals=2,double_value=True)
        self.harris_detector_inputs_container_layout.addWidget(self.harris_detector_block_size_spin_box)
        self.harris_detector_inputs_container_layout.addWidget(self.harris_detector_kernel_size_spin_box)
        self.harris_detector_inputs_container_layout.addWidget(self.harris_detector_k_factor_spin_box)
        self.harris_detector_inputs_container_layout.addWidget(self.harris_detector_threshold_spin_box)


        self.lambda_detector_inputs_container = QWidget()
        self.lambda_detector_inputs_container_layout = QHBoxLayout(self.lambda_detector_inputs_container)
        self.lambda_detector_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.lambda_detector_inputs_container)
        self.lambda_detector_inputs_container.setVisible(False)
        self.lambda_detector_max_corners_spin_box = CustomSpinBox(label="Max Corners",range_start=1,range_end=1000,initial_value=10,step_value=1)
        self.lambda_detector_min_distance_spin_box = CustomSpinBox(label="Min Distance",range_start=1,range_end=1000,initial_value=10,step_value=1)
        self.lambda_detector_quality_level_spin_box = CustomSpinBox(label="Quality Level",range_start=0,range_end=100,initial_value=0.01,step_value=0.01,double_value=True,decimals=2)
        self.lambda_detector_inputs_container_layout.addWidget(self.lambda_detector_max_corners_spin_box)
        self.lambda_detector_inputs_container_layout.addWidget(self.lambda_detector_min_distance_spin_box)
        self.lambda_detector_inputs_container_layout.addWidget(self.lambda_detector_quality_level_spin_box)
        


        self.harris_lambda_detector_inputs_container = QWidget()
        self.harris_lambda_detector_inputs_container_layout = QHBoxLayout(self.harris_lambda_detector_inputs_container)
        self.harris_lambda_detector_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.harris_lambda_detector_inputs_container)
        self.harris_lambda_detector_inputs_container.setVisible(False)
        self.harris_lambda_detector_block_size_spin_box = CustomSpinBox(label="BLock Size",range_start=1,range_end=10,initial_value=2,step_value=1)
        self.harris_lambda_detector_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=11,initial_value=3,step_value=2)
        self.harris_lambda_detector_k_factor_spin_box = CustomSpinBox(label="K Factor",range_start=0,range_end=1000,initial_value=0.04,step_value=0.1,double_value=True,decimals=2)
        self.harris_lambda_detector_threshold_spin_box = CustomSpinBox(label="Threshold",range_start=0,range_end=1000,initial_value=0.01,step_value=0.1,decimals=2,double_value=True)
        self.harris_lambda_detector_max_corners_spin_box = CustomSpinBox(label="Max Corners",range_start=1,range_end=1000,initial_value=10,step_value=1)
        self.harris_lambda_detector_min_distance_spin_box = CustomSpinBox(label="Min Distance",range_start=1,range_end=1000,initial_value=10,step_value=1)
        self.harris_lambda_detector_quality_level_spin_box = CustomSpinBox(label="Quality Level",range_start=0,range_end=100,initial_value=0.01,step_value=0.01,double_value=True,decimals=2)
        
        self.harris_lambda_detector_inputs_container_layout.addWidget(self.harris_lambda_detector_block_size_spin_box)
        self.harris_lambda_detector_inputs_container_layout.addWidget(self.harris_lambda_detector_kernel_size_spin_box)
        self.harris_lambda_detector_inputs_container_layout.addWidget(self.harris_lambda_detector_k_factor_spin_box)
        self.harris_lambda_detector_inputs_container_layout.addWidget(self.harris_lambda_detector_threshold_spin_box)
        self.harris_lambda_detector_inputs_container_layout.addWidget(self.harris_lambda_detector_max_corners_spin_box)
        self.harris_lambda_detector_inputs_container_layout.addWidget(self.harris_lambda_detector_min_distance_spin_box)
        self.harris_lambda_detector_inputs_container_layout.addWidget(self.harris_lambda_detector_quality_level_spin_box)


        self.edge_detection_controller = CornerDetectionController(self)
        
    def on_corner_detector_type_change(self):
        self.hide_all_inputs()

        selected_detector = self.corner_detector_type_custom_combo_box.current_text()

        match selected_detector:
            case "Harris Detector":
                self.harris_detector_inputs_container.setVisible(True)
            case "Lambda Detector":
                self.lambda_detector_inputs_container.setVisible(True)
            case "Harris & Lambda Detector":
                self.harris_lambda_detector_inputs_container.setVisible(True)

    def hide_all_inputs(self):
        self.harris_detector_inputs_container.setVisible(False)
        self.lambda_detector_inputs_container.setVisible(False)
        self.harris_lambda_detector_inputs_container.setVisible(False)

    


