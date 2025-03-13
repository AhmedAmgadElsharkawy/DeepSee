from PyQt5.QtWidgets import QWidget,QHBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_spin_box import CustomSpinBox

from controller.active_contours_cotroller import ActiveContoursController

class ActiveContoursWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(ActiveContoursWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if ActiveContoursWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="Edge Detection")
        ActiveContoursWindow.__instance = self


        self.active_contours_inputs_container = QWidget()
        self.active_contours_inputs_container_layout = QHBoxLayout(self.active_contours_inputs_container)
        self.active_contours_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.active_contours_inputs_container)
        self.active_contours_iterations_spin_box = CustomSpinBox(label="Iterations",range_start=0,range_end=300,initial_value=3,step_value=1)
        self.active_contours_radius_spin_box = CustomSpinBox(label="Radius",range_start=0,range_end=300,initial_value=0,step_value=1)
        self.active_contours_points_spin_box = CustomSpinBox(label="Points",range_start=0,range_end=300,initial_value=0,step_value=1)
        self.active_contours_window_size_spin_box = CustomSpinBox(label="Window Size",range_start=2,range_end=21,initial_value=2,step_value=1)
        self.active_contours_detector_alpha_spin_box = CustomSpinBox(label="Alpha",range_start=0,range_end=100,initial_value=0,step_value=0.1,double_value=True,decimals=1)
        self.active_contours_detector_beta_spin_box = CustomSpinBox(label="Beta",range_start=0,range_end=100,initial_value=0,step_value=0.1,double_value=True,decimals=1)
        self.active_contours_detector_gamma_spin_box = CustomSpinBox(label="Gamma",range_start=0,range_end=100,initial_value=0,step_value=0.1,double_value=True,decimals=1)
        self.active_contours_inputs_container_layout.addWidget(self.active_contours_iterations_spin_box)
        self.active_contours_inputs_container_layout.addWidget(self.active_contours_radius_spin_box)
        self.active_contours_inputs_container_layout.addWidget(self.active_contours_points_spin_box)
        self.active_contours_inputs_container_layout.addWidget(self.active_contours_window_size_spin_box)
        self.active_contours_inputs_container_layout.addWidget(self.active_contours_detector_alpha_spin_box)
        self.active_contours_inputs_container_layout.addWidget(self.active_contours_detector_beta_spin_box)
        self.active_contours_inputs_container_layout.addWidget(self.active_contours_detector_gamma_spin_box)

        self.active_contours_controller = ActiveContoursController(self)
        

    


