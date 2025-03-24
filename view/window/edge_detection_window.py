from PyQt5.QtWidgets import QWidget,QHBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

from controller.edge_detection_controller import EdgeDetectionController

class EdgeDetectionsWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(EdgeDetectionsWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if EdgeDetectionsWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="Edge Detection")
        EdgeDetectionsWindow.__instance = self

        self.edge_detector_type_custom_combo_box = CustomComboBox(label= "Edge Detector Type",combo_box_items_list=["Sobel Detector","Roberts Detector","Prewitt Detector","Canny Detector"])
        self.edge_detector_type_custom_combo_box.currentIndexChanged.connect(self.on_edge_detector_type_change)
        self.inputs_container_layout.addWidget(self.edge_detector_type_custom_combo_box)


        self.sobel_detector_inputs_container = QWidget()
        self.sobel_detector_inputs_container_layout = QHBoxLayout(self.sobel_detector_inputs_container)
        self.sobel_detector_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.sobel_detector_inputs_container)
        self.sobel_detector_direction_custom_combo_box = CustomComboBox(label = "Direction",combo_box_items_list=["Horizontal","Vertical","Combined"])
        self.sobel_detector_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=5,initial_value=3,step_value=2)
        self.sobel_detector_inputs_container_layout.addWidget(self.sobel_detector_direction_custom_combo_box)
        self.sobel_detector_inputs_container_layout.addWidget(self.sobel_detector_kernel_size_spin_box)


        self.roberts_detector_inputs_container = QWidget()
        self.roberts_detector_inputs_container_layout = QHBoxLayout(self.roberts_detector_inputs_container)
        self.roberts_detector_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.roberts_detector_inputs_container)
        self.roberts_detector_inputs_container.setVisible(False)
        self.roberts_detector_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=2,range_end=2,initial_value=2,step_value=0)
        self.roberts_detector_inputs_container_layout.addWidget(self.roberts_detector_kernel_size_spin_box)
        


        self.prewitt_detector_inputs_container = QWidget()
        self.prewitt_detector_inputs_container_layout = QHBoxLayout(self.prewitt_detector_inputs_container)
        self.prewitt_detector_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.prewitt_detector_inputs_container)
        self.prewitt_detector_inputs_container.setVisible(False)
        self.prewitt_detector_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=5,initial_value=3,step_value=2)
        self.prewitt_detector_inputs_container_layout.addWidget(self.prewitt_detector_kernel_size_spin_box)


        self.canny_detector_inputs_container = QWidget()
        self.canny_detector_inputs_container_layout = QHBoxLayout(self.canny_detector_inputs_container)
        self.canny_detector_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.canny_detector_inputs_container)
        self.canny_detector_inputs_container.setVisible(False)
        self.canny_detector_kernel_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=5,initial_value=3,step_value=2)
        self.canny_detector_variance_spin_box = CustomSpinBox(label="Variance",range_start=1,range_end=100,initial_value=1,step_value=1)
        self.canny_detector_lower_threshold_spin_box = CustomSpinBox(label="Lower Threshold",range_start=0,range_end=254,initial_value=0,step_value=1)
        self.canny_detector_upper_threshold_spin_box = CustomSpinBox(label="Upper Threshold",range_start=0,range_end=254,initial_value=0,step_value=1)
        self.canny_detector_inputs_container_layout.addWidget(self.canny_detector_kernel_spin_box)
        self.canny_detector_inputs_container_layout.addWidget(self.canny_detector_variance_spin_box)
        self.canny_detector_inputs_container_layout.addWidget(self.canny_detector_lower_threshold_spin_box)
        self.canny_detector_inputs_container_layout.addWidget(self.canny_detector_upper_threshold_spin_box)

        self.edge_detection_controller = EdgeDetectionController(self)
        
    def on_edge_detector_type_change(self):
        self.hide_all_inputs()

        selected_detector = self.edge_detector_type_custom_combo_box.current_text()

        match selected_detector:
            case "Sobel Detector":
                self.sobel_detector_inputs_container.setVisible(True)
            case "Roberts Detector":
                self.roberts_detector_inputs_container.setVisible(True)
            case "Prewitt Detector":
                self.prewitt_detector_inputs_container.setVisible(True)
            case "Canny Detector":
                self.canny_detector_inputs_container.setVisible(True)

    def hide_all_inputs(self):
        self.sobel_detector_inputs_container.setVisible(False)
        self.roberts_detector_inputs_container.setVisible(False)
        self.prewitt_detector_inputs_container.setVisible(False)
        self.canny_detector_inputs_container.setVisible(False)

    


