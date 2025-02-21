from PyQt5.QtWidgets import QPushButton, QWidget, QVBoxLayout, QLabel, QHBoxLayout

from view.interactive_image_viewer import InteractiveImageViewer
from view.image_viewer import ImageViewer
from view.custom_combo_box import CustomComboBox
from view.custom_spin_box import CustomSpinBox

class ThresholdingWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(0,0,0,0)
        self.main_widget = QWidget(self)
        self.central_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.setSpacing(10)

        self.header_widget = QLabel("Thresholding")
        self.main_widget_layout.addWidget(self.header_widget)

        self.image_viewer_container = QWidget()
        self.image_viewer_container_layout = QHBoxLayout(self.image_viewer_container)
        self.main_widget_layout.addWidget(self.image_viewer_container)
        self.image_viewer_container_layout.setContentsMargins(0,0,0,0)

        self.input_image_viewer = InteractiveImageViewer()
        self.output_image_viewer = ImageViewer()

        self.image_viewer_container_layout.addWidget(self.input_image_viewer)
        self.image_viewer_container_layout.addWidget(self.output_image_viewer)

        self.controls_container = QWidget()
        self.controls_container_layout = QHBoxLayout(self.controls_container)
        self.controls_container_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.addWidget(self.controls_container)

        self.thresholding_type_custom_commbo_box = CustomComboBox(label= "Thresholding Type",combo_box_items_list=["Optimal Thresholding","Otsu Thresholding","Spectral Thresholding"])
        self.controls_container_layout.addWidget(self.thresholding_type_custom_commbo_box)


        self.thresholding_scope_custom_commbo_box = CustomComboBox(label= "Thresholding Scope",combo_box_items_list=["Global Thresholding","Local Thresholding"])
        self.thresholding_scope_custom_commbo_box.currentIndexChanged.connect(self.on_thresholding_scope_change)
        self.controls_container_layout.addWidget(self.thresholding_scope_custom_commbo_box)



        self.local_thresholding_inputs_container = QWidget()
        self.local_thresholding_inputs_container.setVisible(False)
        self.local_thresholding_inputs_container_layout = QHBoxLayout(self.local_thresholding_inputs_container)
        self.local_thresholding_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.local_thresholding_inputs_container)
        self.local_thresholding_window_size_spin_box = CustomSpinBox(label="Window Size",range_start=3,range_end=9,initial_value=3,step_value=1)
        self.local_thresholding_window_offset_spin_box = CustomSpinBox(label="Offset Value",range_start=0,range_end=100,initial_value=0,step_value=1)
        self.local_thresholding_inputs_container_layout.addWidget(self.local_thresholding_window_size_spin_box)
        self.local_thresholding_inputs_container_layout.addWidget(self.local_thresholding_window_offset_spin_box)
        
        
        self.controls_container_layout.addStretch()
        
        self.apply_thresholding_button = QPushButton("Apply")
        self.controls_container_layout.addWidget(self.apply_thresholding_button)
        
    def on_thresholding_scope_change(self):
        selected_scope = self.thresholding_scope_custom_commbo_box.current_text()

        if selected_scope == "Local Thresholding":
            self.local_thresholding_inputs_container.setVisible(True)
        else:
            self.local_thresholding_inputs_container.setVisible(False)


    


