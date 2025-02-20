from PyQt5.QtWidgets import QPushButton, QComboBox, QWidget, QVBoxLayout, QLabel, QHBoxLayout

from view.interactive_image_viewer import InteractiveImageViewer
from view.image_viewer import ImageViewer
from view.custom_combo_box import CustomComboBox

class NoiseWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(0,0,0,0)
        self.main_widget = QWidget(self)
        self.central_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.setSpacing(10)

        self.header_widget = QLabel("Noise")
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

        self.noise_type_custom_commbo_box = CustomComboBox(label= "Noise Type",combo_box_items_list=["option1","option2"])
        self.controls_container_layout.addWidget(self.noise_type_custom_commbo_box)

        

        self.controls_container_layout.addStretch()
        
        self.apply_noise_button = QPushButton("Apply")
        self.controls_container_layout.addWidget(self.apply_noise_button)
        
        

        


        

    


