from PyQt5.QtWidgets import QWidget,QHBoxLayout

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

from controller.segmentation_controller import SegmentationController

class SegmentationWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(SegmentationWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if SegmentationWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="Segmentation")
        SegmentationWindow.__instance = self

        self.segmentation_algorithm_custom_combo_box = CustomComboBox(label= "Segmentation Algorithm",combo_box_items_list=["k-means","Mean Shift","Agglomerative Segmentation","Region Growing"])
        self.segmentation_algorithm_custom_combo_box.currentIndexChanged.connect(self.on_segmentation_algorithm_change)
        self.inputs_container_layout.addWidget(self.segmentation_algorithm_custom_combo_box)


        self.k_means_inputs_container = QWidget()
        self.k_means_inputs_container_layout = QHBoxLayout(self.k_means_inputs_container)
        self.k_means_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.k_means_inputs_container)
        self.k_means_k_value_spin_box = CustomSpinBox(label="K Value",range_start=1,range_end=20,initial_value=2,step_value=1)
        self.k_means_max_iterations_spin_box = CustomSpinBox(label="Max Iterations",range_start=1,range_end=100,initial_value=50,step_value=1)
        self.k_means_inputs_container_layout.addWidget(self.k_means_k_value_spin_box)
        self.k_means_inputs_container_layout.addWidget(self.k_means_max_iterations_spin_box)


        self.mean_shift_inputs_container = QWidget()
        self.mean_shift_inputs_container_layout = QHBoxLayout(self.mean_shift_inputs_container)
        self.mean_shift_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.mean_shift_inputs_container)
        self.mean_shift_inputs_container.setVisible(False)
        self.mean_shift_spatial_radius_spin_box = CustomSpinBox(label="Spatial Radius",range_start=1,range_end=100,initial_value=15,step_value=1)
        self.mean_shift_color_radius_spin_box = CustomSpinBox(label="Color Radius",range_start=1,range_end=100,initial_value=20,step_value=1)
        self.mean_shift_inputs_container_layout.addWidget(self.mean_shift_spatial_radius_spin_box)
        self.mean_shift_inputs_container_layout.addWidget(self.mean_shift_color_radius_spin_box)
        


        self.agglomerative_segmentation_inputs_container = QWidget()
        self.agglomerative_segmentation_inputs_container_layout = QHBoxLayout(self.agglomerative_segmentation_inputs_container)
        self.agglomerative_segmentation_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.agglomerative_segmentation_inputs_container)
        self.agglomerative_segmentation_inputs_container.setVisible(False)
        self.agglomerative_segmentation_clusters_number_spin_box = CustomSpinBox(label="Clusters Number",range_start=1,range_end=25,initial_value=10,step_value=1)
        self.agglomerative_segmentation_initial_clusters_number_spin_box = CustomSpinBox(label="Initial Clusters Number",range_start=1,range_end=100,initial_value=25,step_value=1)
        self.agglomerative_segmentation_inputs_container_layout.addWidget(self.agglomerative_segmentation_clusters_number_spin_box)
        self.agglomerative_segmentation_inputs_container_layout.addWidget(self.agglomerative_segmentation_initial_clusters_number_spin_box)
        self.agglomerative_segmentation_initial_clusters_number_spin_box.valueChanged.connect(self.update_final_clusters_range)



        self.region_growing_inputs_container = QWidget()
        self.region_growing_inputs_container_layout = QHBoxLayout(self.region_growing_inputs_container)
        self.region_growing_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.region_growing_inputs_container)
        self.region_growing_inputs_container.setVisible(False)
        self.region_growing_threshold_spin_box = CustomSpinBox(label="Threshold",range_start=1,range_end=100,initial_value=2,step_value=1)
        self.region_growing_inputs_container_layout.addWidget(self.region_growing_threshold_spin_box)

        self.edge_detection_controller = SegmentationController(self)

        
    def on_segmentation_algorithm_change(self):
        self.hide_all_inputs()

        selected_detector = self.segmentation_algorithm_custom_combo_box.current_text()

        match selected_detector:
            case "k-means":
                self.k_means_inputs_container.setVisible(True)
                self.input_image_viewer.enable_add_marker(False)
            case "Mean Shift":
                self.mean_shift_inputs_container.setVisible(True)
                self.input_image_viewer.enable_add_marker(False)
            case "Agglomerative Segmentation":
                self.agglomerative_segmentation_inputs_container.setVisible(True)
                self.input_image_viewer.enable_add_marker(False)
            case "Region Growing":
                self.region_growing_inputs_container.setVisible(True)
                self.input_image_viewer.enable_add_marker(True)

    def hide_all_inputs(self):
        self.k_means_inputs_container.setVisible(False)
        self.mean_shift_inputs_container.setVisible(False)
        self.agglomerative_segmentation_inputs_container.setVisible(False)
        self.region_growing_inputs_container.setVisible(False)


    def update_final_clusters_range(self, value):
        """
        Updates the maximum value of the final clusters spin box
        based on the initial clusters spin box value.
        """
        self.agglomerative_segmentation_clusters_number_spin_box.setMaximum(value)
        # Optionally, also fix current value if it became invalid
        if self.agglomerative_segmentation_clusters_number_spin_box.value() > value:
            self.agglomerative_segmentation_clusters_number_spin_box.setValue(value)


    


