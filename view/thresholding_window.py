from PyQt5.QtWidgets import QWidget, QHBoxLayout

from view.basic_stacked_window import BasicStackedWindow
from view.custom_combo_box import CustomComboBox
from view.custom_spin_box import CustomSpinBox

class ThresholdingWindow(BasicStackedWindow):
    def __init__(self):
        super().__init__()

        self.thresholding_type_custom_combo_box = CustomComboBox(label= "Thresholding Type",combo_box_items_list=["Optimal Thresholding","Otsu Thresholding","Spectral Thresholding"])
        self.controls_container_layout.addWidget(self.thresholding_type_custom_combo_box)

        self.thresholding_scope_custom_combo_box = CustomComboBox(label= "Thresholding Scope",combo_box_items_list=["Global Thresholding","Local Thresholding"])
        self.thresholding_scope_custom_combo_box.currentIndexChanged.connect(self.on_thresholding_scope_change)
        self.controls_container_layout.addWidget(self.thresholding_scope_custom_combo_box)

        self.local_thresholding_inputs_container = QWidget()
        self.local_thresholding_inputs_container.setVisible(False)
        self.local_thresholding_inputs_container_layout = QHBoxLayout(self.local_thresholding_inputs_container)
        self.local_thresholding_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.local_thresholding_inputs_container)
        self.local_thresholding_window_size_spin_box = CustomSpinBox(label="Window Size",range_start=3,range_end=9,initial_value=3,step_value=1)
        self.local_thresholding_window_offset_spin_box = CustomSpinBox(label="Offset Value",range_start=0,range_end=100,initial_value=0,step_value=1)
        self.local_thresholding_inputs_container_layout.addWidget(self.local_thresholding_window_size_spin_box)
        self.local_thresholding_inputs_container_layout.addWidget(self.local_thresholding_window_offset_spin_box)
        
        
    def on_thresholding_scope_change(self):
        selected_scope = self.thresholding_scope_custom_combo_box.current_text()

        if selected_scope == "Local Thresholding":
            self.local_thresholding_inputs_container.setVisible(True)
        else:
            self.local_thresholding_inputs_container.setVisible(False)


    


