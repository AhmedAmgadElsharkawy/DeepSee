from PyQt5.QtWidgets import QWidget, QHBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

from controller.thresholding_controller import ThresholdingController

class ThresholdingWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(ThresholdingWindow, cls).__new__(cls)
        return cls.__instance    

    def __init__(self, main_window):
        if ThresholdingWindow.__instance != None:
            return
        
        super().__init__(main_window, "Thresholding")
        ThresholdingWindow.__instance =self

        self.thresholding_scope_custom_combo_box = CustomComboBox(label= "Thresholding Scope",combo_box_items_list=["Global Thresholding","Local Thresholding"])
        self.thresholding_scope_custom_combo_box.currentIndexChanged.connect(self.on_thresholding_scope_change)
        self.inputs_container_layout.addWidget(self.thresholding_scope_custom_combo_box)
        self.thresholding_type_custom_combo_box = CustomComboBox(label= "Thresholding Type",combo_box_items_list=["Otsu Thresholding", "Global Mean"])
        self.inputs_container_layout.addWidget(self.thresholding_type_custom_combo_box)

        self.local_thresholding_inputs_container = QWidget()
        self.local_thresholding_inputs_container.setVisible(False)
        self.local_thresholding_inputs_container_layout = QHBoxLayout(self.local_thresholding_inputs_container)
        self.local_thresholding_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.local_thresholding_inputs_container)
        self.local_thresholding_window_size_spin_box = CustomSpinBox(label="Window Size",range_start=3,range_end=11,initial_value=11,step_value=2)
        self.local_thresholding_window_offset_spin_box = CustomSpinBox(label="Offset Value",range_start=-20,range_end=20,initial_value=2,step_value=1)
        self.local_thresholding_inputs_container_layout.addWidget(self.local_thresholding_window_size_spin_box)
        self.local_thresholding_inputs_container_layout.addWidget(self.local_thresholding_window_offset_spin_box)
        self.variance_spin_box = CustomSpinBox(
            label="Variance", range_start=1, range_end=10, initial_value=1, step_value=1
        )
        self.variance_spin_box.setVisible(False)
        self.inputs_container_layout.addWidget(self.variance_spin_box)
        self.thresholding_type_custom_combo_box.currentIndexChanged.connect(self.on_thresholding_type_change)

        self.thresholding_controller = ThresholdingController(self)
        
        
    def on_thresholding_scope_change(self):
        selected_scope = self.thresholding_scope_custom_combo_box.current_text()

        if selected_scope == "Local Thresholding":
            self.local_thresholding_inputs_container.setVisible(True)
            new_items = ["Adaptive Mean", "Adaptive Gaussian"]
        else:
            self.local_thresholding_inputs_container.setVisible(False)
            new_items = ["Otsu Thresholding", "Global Mean"]


        self.thresholding_type_custom_combo_box.clear_iteams()
        self.thresholding_type_custom_combo_box.add_item(new_items)

    def on_thresholding_type_change(self):
            selected_type = self.thresholding_type_custom_combo_box.current_text()
            if selected_type == "Adaptive Gaussian":
                self.variance_spin_box.setVisible(True)
            else:
                self.variance_spin_box.setVisible(False)


    


