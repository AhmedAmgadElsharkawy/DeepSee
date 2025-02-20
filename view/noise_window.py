from PyQt5.QtWidgets import QPushButton, QWidget, QVBoxLayout, QLabel, QHBoxLayout

from view.interactive_image_viewer import InteractiveImageViewer
from view.image_viewer import ImageViewer
from view.custom_combo_box import CustomComboBox
from view.custom_spin_box import CustomSpinBox

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

        self.noise_type_custom_commbo_box = CustomComboBox(label= "Noise Type",combo_box_items_list=["Uniform Noise","Gaussian Noise","Salt & Pepper Noise"])
        self.noise_type_custom_commbo_box.currentIndexChanged.connect(self.on_noise_type_change)
        self.controls_container_layout.addWidget(self.noise_type_custom_commbo_box)


        self.uniform_nosie_inputs_container = QWidget()
        self.uniform_nosie_inputs_container_layout = QHBoxLayout(self.uniform_nosie_inputs_container)
        self.uniform_nosie_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.uniform_nosie_inputs_container)
        self.noise_value_spin_box = CustomSpinBox(label="Noise Value",range_start=0,range_end=1000,initial_value=0,step_value=1)
        self.uniform_nosie_inputs_container_layout.addWidget(self.noise_value_spin_box)


        self.gaussian_nosie_inputs_container = QWidget()
        self.gaussian_nosie_inputs_container_layout = QHBoxLayout(self.gaussian_nosie_inputs_container)
        self.gaussian_nosie_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.gaussian_nosie_inputs_container)
        self.gaussian_nosie_inputs_container.setVisible(False)
        self.guassian_noise_mean_spin_box = CustomSpinBox(label="Mean")
        self.guassian_noise_variance_spin_box  = CustomSpinBox(label="Variance")
        self.gaussian_nosie_inputs_container_layout.addWidget(self.guassian_noise_mean_spin_box)
        self.gaussian_nosie_inputs_container_layout.addWidget(self.guassian_noise_variance_spin_box)
        


        self.salt_and_pepper_nosie_inputs_container = QWidget()
        self.salt_and_pepper_nosie_inputs_container_layout = QHBoxLayout(self.salt_and_pepper_nosie_inputs_container)
        self.salt_and_pepper_nosie_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.salt_and_pepper_nosie_inputs_container)
        self.salt_and_pepper_nosie_inputs_container.setVisible(False)
        self.salt_probability_spin_box = CustomSpinBox(label="Salt Probability")
        self.pepper_probability_spin_box  = CustomSpinBox(label="Pepper Probability")
        self.salt_and_pepper_nosie_inputs_container_layout.addWidget(self.salt_probability_spin_box)
        self.salt_and_pepper_nosie_inputs_container_layout.addWidget(self.pepper_probability_spin_box)

        self.controls_container_layout.addStretch()
        
        self.apply_noise_button = QPushButton("Apply")
        self.controls_container_layout.addWidget(self.apply_noise_button)
        
    def on_noise_type_change(self):
        if self.noise_type_custom_commbo_box.current_text() == "Uniform Noise":
            self.uniform_nosie_inputs_container.setVisible(True)
            self.salt_and_pepper_nosie_inputs_container.setVisible(False)
            self.gaussian_nosie_inputs_container.setVisible(False)

        elif self.noise_type_custom_commbo_box.current_text() == "Gaussian Noise":
            self.gaussian_nosie_inputs_container.setVisible(True)
            self.salt_and_pepper_nosie_inputs_container.setVisible(False)
            self.uniform_nosie_inputs_container.setVisible(False)

        else:
            self.salt_and_pepper_nosie_inputs_container.setVisible(True)
            self.gaussian_nosie_inputs_container.setVisible(False)
            self.uniform_nosie_inputs_container.setVisible(False)


        


        

    


