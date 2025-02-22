from PyQt5.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox
from view.widget.interactive_image_viewer import InteractiveImageViewer
from view.widget.image_viewer import ImageViewer

from controller.hybrid_image_controller import HybridImageController

class HybridImageWindow(BasicStackedWindow):
    def __init__(self):
        super().__init__(header_text="Hybrid Image")

        self.image_viewers_container.deleteLater()

        self.first_image_filter_type_custom_combo_box = CustomComboBox(label= "First Image Filter",combo_box_items_list=["Low Pass Filter","High Pass Filter"])
        self.inputs_container_layout.addWidget(self.first_image_filter_type_custom_combo_box)

        self.second_image_filter_type_custom_combo_box = CustomComboBox(label= "Second Image Filter",combo_box_items_list=["Low Pass Filter","High Pass Filter"])
        self.inputs_container_layout.addWidget(self.second_image_filter_type_custom_combo_box)

        self.radius_custom_spin_box = CustomSpinBox(label= "Filter Radius",range_start=0,range_end=10,initial_value=0)
        self.inputs_container_layout.addWidget(self.radius_custom_spin_box)

        self.image_viewers_container = QWidget()
        self.image_viewers_container_layout = QHBoxLayout(self.image_viewers_container)
        self.image_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.addWidget(self.image_viewers_container)

        self.input_images_viewers_container = QWidget()
        self.input_images_viewers_container_layout = QVBoxLayout(self.input_images_viewers_container)
        self.input_images_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.image_viewers_container_layout.addWidget(self.input_images_viewers_container)

        self.first_image_viewers_container = QWidget()
        self.first_image_viewers_container_layout = QHBoxLayout(self.first_image_viewers_container)
        self.first_image_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.input_images_viewers_container_layout.addWidget(self.first_image_viewers_container)
        self.first_original_image_viewer = InteractiveImageViewer()
        self.first_filtered_image_viewer = ImageViewer()
        self.first_image_viewers_container_layout.addWidget(self.first_original_image_viewer)
        self.first_image_viewers_container_layout.addWidget(self.first_filtered_image_viewer)

        self.second_image_viewers_container = QWidget()
        self.second_image_viewers_container_layout = QHBoxLayout(self.second_image_viewers_container)
        self.second_image_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.input_images_viewers_container_layout.addWidget(self.second_image_viewers_container)
        self.second_original_image_viewer = InteractiveImageViewer()
        self.second_filtered_image_viewer = ImageViewer()
        self.second_image_viewers_container_layout.addWidget(self.second_original_image_viewer)
        self.second_image_viewers_container_layout.addWidget(self.second_filtered_image_viewer)

        self.hybrid_image_viewer = ImageViewer()
        self.image_viewers_container_layout.addWidget(self.hybrid_image_viewer)

        self.hybrid_image_controller = HybridImageController(self)


        


        

    


