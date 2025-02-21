from PyQt5.QtWidgets import QPushButton, QWidget, QVBoxLayout, QLabel, QHBoxLayout

from view.interactive_image_viewer import InteractiveImageViewer
from view.image_viewer import ImageViewer
from view.custom_combo_box import CustomComboBox
from view.custom_spin_box import CustomSpinBox

class FiltersWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(0,0,0,0)
        self.main_widget = QWidget(self)
        self.central_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.setSpacing(10)

        self.header_widget = QLabel("Filters")
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

        self.filter_type_custom_commbo_box = CustomComboBox(label= "Filter Type",combo_box_items_list=["Average Filter","Gaussian Filter","Median Filter","Low Pass Filter","High Pass Filter"])
        self.filter_type_custom_commbo_box.currentIndexChanged.connect(self.on_filter_type_change)
        self.controls_container_layout.addWidget(self.filter_type_custom_commbo_box)


        self.average_filter_inputs_container = QWidget()
        self.average_filter_inputs_container_layout = QHBoxLayout(self.average_filter_inputs_container)
        self.average_filter_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.average_filter_inputs_container)
        self.average_filter_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=9,initial_value=3,step_value=1)
        self.average_filter_inputs_container_layout.addWidget(self.average_filter_kernel_size_spin_box)


        self.gaussian_filter_inputs_container = QWidget()
        self.gaussian_filter_inputs_container_layout = QHBoxLayout(self.gaussian_filter_inputs_container)
        self.gaussian_filter_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.gaussian_filter_inputs_container)
        self.gaussian_filter_inputs_container.setVisible(False)
        self.guassian_filter_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=9,initial_value=3,step_value=1)
        self.guassian_filter_variance_spin_box  = CustomSpinBox(label="Variance")
        self.gaussian_filter_inputs_container_layout.addWidget(self.guassian_filter_kernel_size_spin_box)
        self.gaussian_filter_inputs_container_layout.addWidget(self.guassian_filter_variance_spin_box)
        


        self.median_filter_inputs_container = QWidget()
        self.median_filter_inputs_container_layout = QHBoxLayout(self.median_filter_inputs_container)
        self.median_filter_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.median_filter_inputs_container)
        self.median_filter_inputs_container.setVisible(False)
        self.median_filter_kernel_size_spin_box = CustomSpinBox(label="Kernel Size",range_start=3,range_end=9,initial_value=3,step_value=1)
        self.median_filter_inputs_container_layout.addWidget(self.median_filter_kernel_size_spin_box)


        self.low_pass_filter_inputs_container = QWidget()
        self.low_pass_filter_inputs_container_layout = QHBoxLayout(self.low_pass_filter_inputs_container)
        self.low_pass_filter_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.low_pass_filter_inputs_container)
        self.low_pass_filter_inputs_container.setVisible(False)
        self.low_pass_filter_radius_spin_box = CustomSpinBox(label="Radius",range_start=3,range_end=9,initial_value=3,step_value=1)
        self.low_pass_filter_inputs_container_layout.addWidget(self.low_pass_filter_radius_spin_box)

        self.high_pass_filter_inputs_container = QWidget()
        self.high_pass_filter_inputs_container_layout = QHBoxLayout(self.high_pass_filter_inputs_container)
        self.high_pass_filter_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.high_pass_filter_inputs_container)
        self.high_pass_filter_inputs_container.setVisible(False)
        self.high_pass_filter_radius_spin_box = CustomSpinBox(label="Radius",range_start=3,range_end=9,initial_value=3,step_value=1)
        self.high_pass_filter_inputs_container_layout.addWidget(self.high_pass_filter_radius_spin_box)


        self.controls_container_layout.addStretch()
        
        self.apply_filter_button = QPushButton("Apply")
        self.controls_container_layout.addWidget(self.apply_filter_button)
        
    def on_filter_type_change(self):
        self.hide_all_inputs()

        selected_filter = self.filter_type_custom_commbo_box.current_text()

        match selected_filter:
            case "Average Filter":
                self.average_filter_inputs_container.setVisible(True)
            case "Gaussian Filter":
                self.gaussian_filter_inputs_container.setVisible(True)
            case "Median Filter":
                self.median_filter_inputs_container.setVisible(True)
            case "Low Pass Filter":
                self.low_pass_filter_inputs_container.setVisible(True)
            case "High Pass Filter":
                self.high_pass_filter_inputs_container.setVisible(True)
    
    def hide_all_inputs(self):
        self.high_pass_filter_inputs_container.setVisible(False)
        self.low_pass_filter_inputs_container.setVisible(False)
        self.median_filter_inputs_container.setVisible(False)
        self.gaussian_filter_inputs_container.setVisible(False)
        self.average_filter_inputs_container.setVisible(False)
    


