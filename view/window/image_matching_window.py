from PyQt5.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout
from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.interactive_image_viewer import InteractiveImageViewer
from view.widget.image_viewer import ImageViewer

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox

from controller.image_matching_controller import ImageMatchingController


class ImageMatchingWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(ImageMatchingWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if ImageMatchingWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="Image Matching")
        ImageMatchingWindow.__instance = self

        self.matching_algorithm_custom_combo_box = CustomComboBox(label= "Matching Algorithm",combo_box_items_list=["Sum Of Squared Differences","Normalized Cross Correlation"])
        self.inputs_container_layout.addWidget(self.matching_algorithm_custom_combo_box)
        
        self.image_viewers_container.deleteLater()


        self.image_viewers_container = QWidget()
        self.image_viewers_container_layout = QHBoxLayout(self.image_viewers_container)
        self.image_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.addWidget(self.image_viewers_container)

        self.input_images_viewers_container = QWidget()
        self.input_images_viewers_container_layout = QVBoxLayout(self.input_images_viewers_container)
        self.input_images_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.image_viewers_container_layout.addWidget(self.input_images_viewers_container)


        self.input_template_image_viewer = InteractiveImageViewer(custom_placeholder="Double click, or drop the image here\n\nAllowed Files: PNG, JPG, JPEG BMP files")
        self.input_images_viewers_container_layout.addWidget(self.input_template_image_viewer)

        self.input_template_image_viewer = InteractiveImageViewer(custom_placeholder="Double click, or drop the template here\n\nAllowed Files: PNG, JPG, JPEG BMP files")
        self.input_images_viewers_container_layout.addWidget(self.input_template_image_viewer)

        self.output_image_viewer = ImageViewer(custom_placeholder = "Matching result will appear here")
        self.image_viewers_container_layout.addWidget(self.output_image_viewer)

        self.image_matching_controller = ImageMatchingController(self)


