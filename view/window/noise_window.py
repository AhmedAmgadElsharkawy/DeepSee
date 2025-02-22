from PyQt5.QtWidgets import QWidget, QHBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

from controller.noise_controller import NoiseController

class NoiseWindow(BasicStackedWindow):
    def __init__(self):
        super().__init__("Noise")

        self.noise_type_custom_combo_box = CustomComboBox(label= "Noise Type",combo_box_items_list=["Uniform Noise","Gaussian Noise","Salt & Pepper Noise"])
        self.noise_type_custom_combo_box.currentIndexChanged.connect(self.on_noise_type_change)
        self.inputs_container_layout.addWidget(self.noise_type_custom_combo_box)


        self.uniform_nosie_inputs_container = QWidget()
        self.uniform_nosie_inputs_container_layout = QHBoxLayout(self.uniform_nosie_inputs_container)
        self.uniform_nosie_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.uniform_nosie_inputs_container)
        self.noise_value_spin_box = CustomSpinBox(label="Noise Value",range_start=0,range_end=1000,initial_value=0,step_value=1)
        self.uniform_nosie_inputs_container_layout.addWidget(self.noise_value_spin_box)


        self.gaussian_nosie_inputs_container = QWidget()
        self.gaussian_nosie_inputs_container_layout = QHBoxLayout(self.gaussian_nosie_inputs_container)
        self.gaussian_nosie_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.gaussian_nosie_inputs_container)
        self.gaussian_nosie_inputs_container.setVisible(False)
        self.guassian_noise_mean_spin_box = CustomSpinBox(label="Mean")
        self.guassian_noise_variance_spin_box  = CustomSpinBox(label="Variance")
        self.gaussian_nosie_inputs_container_layout.addWidget(self.guassian_noise_mean_spin_box)
        self.gaussian_nosie_inputs_container_layout.addWidget(self.guassian_noise_variance_spin_box)
        


        self.salt_and_pepper_nosie_inputs_container = QWidget()
        self.salt_and_pepper_nosie_inputs_container_layout = QHBoxLayout(self.salt_and_pepper_nosie_inputs_container)
        self.salt_and_pepper_nosie_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.salt_and_pepper_nosie_inputs_container)
        self.salt_and_pepper_nosie_inputs_container.setVisible(False)
        self.salt_probability_spin_box = CustomSpinBox(label="Salt Probability")
        self.pepper_probability_spin_box  = CustomSpinBox(label="Pepper Probability")
        self.salt_and_pepper_nosie_inputs_container_layout.addWidget(self.salt_probability_spin_box)
        self.salt_and_pepper_nosie_inputs_container_layout.addWidget(self.pepper_probability_spin_box)

        self.noise_controller = NoiseController(self)

    def on_noise_type_change(self):
        self.hide_all_inputs()

        selected_noise = self.noise_type_custom_combo_box.current_text()

        match selected_noise:
            case "Uniform Noise":
                self.uniform_nosie_inputs_container.setVisible(True)
            case "Gaussian Noise":
                self.gaussian_nosie_inputs_container.setVisible(True)
            case "Salt & Pepper Noise":
                self.salt_and_pepper_nosie_inputs_container.setVisible(True)

    def hide_all_inputs(self):
        self.uniform_nosie_inputs_container.setVisible(False)
        self.gaussian_nosie_inputs_container.setVisible(False)
        self.salt_and_pepper_nosie_inputs_container.setVisible(False)



        


        

    


