from PyQt5.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout,QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.interactive_image_viewer import InteractiveImageViewer
from view.widget.image_viewer import ImageViewer

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

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


        self.input_image_viewer = InteractiveImageViewer(custom_placeholder="Double click, or drop the image here\n\nAllowed Files: PNG, JPG, JPEG BMP files")
        self.input_images_viewers_container_layout.addWidget(self.input_image_viewer)

        self.input_template_viewer = InteractiveImageViewer(custom_placeholder="Double click, or drop the template here\n\nAllowed Files: PNG, JPG, JPEG BMP files")
        self.input_images_viewers_container_layout.addWidget(self.input_template_viewer)

        self.output_image_viewer = ImageViewer(custom_placeholder = "Matching result will appear here")
        self.image_viewers_container_layout.addWidget(self.output_image_viewer)


        self.img_detect_keypoints_inputs_container = QWidget()
        self.img_detect_keypoints_inputs_container_layout = QHBoxLayout(self.img_detect_keypoints_inputs_container)
        self.img_detect_keypoints_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.img_detect_keypoints_inputs_container)
        self.img_detect_keypoints_sigma_spin_box = CustomSpinBox(label="Img Sigma",range_start=0.1,range_end=100,initial_value=1.6,step_value=0.01,decimals=2,double_value=True)
        self.img_detect_keypoints_assumed_blur_spin_box = CustomSpinBox(label="Img Assumed Blur",range_start=0.01,range_end=100,initial_value=0.5,step_value=0.01,decimals=2,double_value=True)
        self.img_detect_keypoints_intervals_number_spin_box = CustomSpinBox(label="Img Intervals Number",range_start=1,range_end=5,initial_value=3,step_value=1)
        self.img_detect_keypoints_inputs_container_layout.addWidget(self.img_detect_keypoints_sigma_spin_box)
        self.img_detect_keypoints_inputs_container_layout.addWidget(self.img_detect_keypoints_assumed_blur_spin_box)
        self.img_detect_keypoints_inputs_container_layout.addWidget(self.img_detect_keypoints_intervals_number_spin_box)



        self.template_detect_keypoints_inputs_container = QWidget()
        self.template_detect_keypoints_inputs_container_layout = QHBoxLayout(self.template_detect_keypoints_inputs_container)
        self.template_detect_keypoints_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.template_detect_keypoints_inputs_container)
        self.template_detect_keypoints_sigma_spin_box = CustomSpinBox(label="Template Sigma",range_start=0.1,range_end=100,initial_value=1.6,step_value=0.01,decimals=2,double_value=True)
        self.template_detect_keypoints_assumed_blur_spin_box = CustomSpinBox(label="Template Assumed Blur",range_start=0.01,range_end=100,initial_value=0.5,step_value=0.01,decimals=2,double_value=True)
        self.template_detect_keypoints_intervals_number_spin_box = CustomSpinBox(label="Template Intervals Number",range_start=1,range_end=5,initial_value=3,step_value=1)
        self.template_detect_keypoints_inputs_container_layout.addWidget(self.template_detect_keypoints_sigma_spin_box)
        self.template_detect_keypoints_inputs_container_layout.addWidget(self.template_detect_keypoints_assumed_blur_spin_box)
        self.template_detect_keypoints_inputs_container_layout.addWidget(self.template_detect_keypoints_intervals_number_spin_box)

        self.image_matching_controller = ImageMatchingController(self)

        self.time_elapsed_container =  QWidget()
        self.time_elapsed_container_layout = QHBoxLayout(self.time_elapsed_container)
        self.time_elapsed_container_layout.setContentsMargins(0,0,0,0)
        self.time_elapsed_title = QLabel("Time Elapsed:")
        # font = QFont("Inter", 11)
        # font.setWeight(QFont.Medium)
        self.time_elapsed_value = QLabel("-- Seconds")
        # self.time_elapsed_value.setFont(font)
        # self.time_elapsed_title.setFont(font)
        self.time_elapsed_title.setObjectName("time_elapsed_title")
        self.time_elapsed_value.setObjectName("time_elapsed_value")
        self.time_elapsed_container_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.time_elapsed_container_layout.addWidget(self.time_elapsed_title)
        self.time_elapsed_container_layout.addWidget(self.time_elapsed_value)
        self.main_widget_layout.addWidget(self.time_elapsed_container)
        self.time_elapsed_container_layout.setSpacing(20)
        
        


