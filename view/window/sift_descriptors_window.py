from PyQt5.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox
from view.widget.interactive_image_viewer import InteractiveImageViewer
from view.widget.image_viewer import ImageViewer


from controller.sift_descriptors_controller import SiftDescriptorsController

class SiftDescriptorsWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(SiftDescriptorsWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if SiftDescriptorsWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="SIFT Descriptors")
        SiftDescriptorsWindow.__instance = self

        self.sift_mode_custom_combo_box = CustomComboBox(label= "SIFT Mode",combo_box_items_list=["Detect Keypoints","Match Features"])
        self.sift_mode_custom_combo_box.currentIndexChanged.connect(self.on_sift_mode_change)
        self.inputs_container_layout.addWidget(self.sift_mode_custom_combo_box)


        self.image_viewers_container.deleteLater()


        self.detect_keypoints_inputs_container = QWidget()
        self.detect_keypoints_inputs_container_layout = QHBoxLayout(self.detect_keypoints_inputs_container)
        self.detect_keypoints_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.detect_keypoints_inputs_container)
        self.detect_keypoints_sigma_spin_box = CustomSpinBox(label="Sigma",range_start=0.1,range_end=100,initial_value=0.1,step_value=0.01,decimals=2,double_value=True)
        self.detect_keypoints_assumed_blur_spin_box = CustomSpinBox(label="Assumed Blur",range_start=0.01,range_end=100,initial_value=0.01,step_value=0.01,decimals=2,double_value=True)
        self.detect_keypoints_intervals_number_spin_box = CustomSpinBox(label="Intervals Number",range_start=1,range_end=5,initial_value=1,step_value=1)
        self.detect_keypoints_inputs_container_layout.addWidget(self.detect_keypoints_sigma_spin_box)
        self.detect_keypoints_inputs_container_layout.addWidget(self.detect_keypoints_assumed_blur_spin_box)
        self.detect_keypoints_inputs_container_layout.addWidget(self.detect_keypoints_intervals_number_spin_box)
        


        self.match_features_inputs_container = QWidget()
        self.match_features_inputs_container_layout = QHBoxLayout(self.match_features_inputs_container)
        self.match_features_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.match_features_inputs_container)
        self.match_features_inputs_container.setVisible(False)
        self.match_features_algorithm_combo_box = CustomComboBox(label= "Matching Algorithm",combo_box_items_list=["Sum Of Squared Differences","Normalized Cross Correlation"])
        self.match_features_matches_number_spin_box = CustomSpinBox(label="Matches Number",range_start=1,range_end=100,initial_value=1,step_value=1)
        self.match_features_inputs_container_layout.addWidget(self.match_features_algorithm_combo_box)
        self.match_features_inputs_container_layout.addWidget(self.match_features_matches_number_spin_box)


        self.detect_keypoints_image_viewers_container = QWidget()
        self.detect_keypoints_image_viewers_container_layout = QHBoxLayout(self.detect_keypoints_image_viewers_container)
        self.detect_keypoints_image_viewers_container_layout.setContentsMargins(0, 0, 0, 0)
        self.detect_keypoints_image_viewers_container_layout.setSpacing(10)

        self.detect_keypoints_input_image_viewer = InteractiveImageViewer()
        self.detect_keypoints_output_image_viewer = ImageViewer()

        self.detect_keypoints_image_viewers_container_layout.addWidget(self.detect_keypoints_input_image_viewer)
        self.detect_keypoints_image_viewers_container_layout.addWidget(self.detect_keypoints_output_image_viewer)


        
        self.match_features_image_viewers_container = QWidget()
        self.match_features_image_viewers_container_layout = QHBoxLayout(self.match_features_image_viewers_container)
        self.match_features_image_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.addWidget(self.match_features_image_viewers_container)

        self.match_features_input_images_viewers_container = QWidget()
        self.match_features_input_images_viewers_container_layout = QVBoxLayout(self.match_features_input_images_viewers_container)
        self.match_features_input_images_viewers_container_layout.setContentsMargins(0,0,0,0)
        self.match_features_image_viewers_container_layout.addWidget(self.match_features_input_images_viewers_container)

        self.matching_features_input_image_viewer = InteractiveImageViewer(custom_placeholder="Double click, or drop the image here\n\nAllowed Files: PNG, JPG, JPEG BMP files")
        self.match_features_input_images_viewers_container_layout.addWidget(self.matching_features_input_image_viewer)
        self.mathcing_features_input_template_viewer = InteractiveImageViewer(custom_placeholder="Double click, or drop the template here\n\nAllowed Files: PNG, JPG, JPEG BMP files")
        self.match_features_input_images_viewers_container_layout.addWidget(self.mathcing_features_input_template_viewer)
        self.matching_output_image_viewer = ImageViewer(custom_placeholder = "Matching result will appear here")
        self.match_features_image_viewers_container_layout.addWidget(self.matching_output_image_viewer)

        self.main_widget_layout.addWidget(self.detect_keypoints_image_viewers_container)
        self.main_widget_layout.addWidget(self.match_features_image_viewers_container)
        self.match_features_image_viewers_container.setVisible(False)

        self.sift_controller = SiftDescriptorsController(self)
        
    def on_sift_mode_change(self):
        self.hide_all_inputs()

        selected_detector = self.sift_mode_custom_combo_box.current_text()

        match selected_detector:
            case "Detect Keypoints":
                self.detect_keypoints_image_viewers_container.setVisible(True)
                self.detect_keypoints_inputs_container.setVisible(True)
            case "Match Features":
                self.match_features_image_viewers_container.setVisible(True)
                self.match_features_inputs_container.setVisible(True)

    def hide_all_inputs(self):
        self.detect_keypoints_image_viewers_container.setVisible(False)
        self.match_features_image_viewers_container.setVisible(False)
        self.detect_keypoints_inputs_container.setVisible(False)
        self.match_features_inputs_container.setVisible(False)
        

    


