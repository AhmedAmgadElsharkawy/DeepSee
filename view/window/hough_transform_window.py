from PyQt5.QtWidgets import QWidget,QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QColorDialog
from PyQt5.QtGui import QColor, QPixmap,QIcon,QFont
from PyQt5.QtCore import QSize,Qt


from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.custom_spin_box import CustomSpinBox

from controller.hough_transform_controller import HoughTransformController

class HoughTransformWindow(BasicStackedWindow):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(HoughTransformWindow, cls).__new__(cls)
        return cls.__instance    
    
    def __init__(self, main_window):
        if HoughTransformWindow.__instance != None:
            return
        
        super().__init__(main_window, header_text="Edge Detection")
        HoughTransformWindow.__instance = self

        self.detected_objects_type_custom_combo_box = CustomComboBox(label= "Objects Type",combo_box_items_list=["Lines Detection","Circles Detection","Ellipses Detection"])
        self.detected_objects_type_custom_combo_box.currentIndexChanged.connect(self.on_detected_objects_type_change)
        self.inputs_container_layout.addWidget(self.detected_objects_type_custom_combo_box)


        self.linear_detection_inputs_container = QWidget()
        self.linear_detection_inputs_container_layout = QHBoxLayout(self.linear_detection_inputs_container)
        self.linear_detection_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.linear_detection_inputs_container)

        self.linear_detection_theta_custom_spin_box = CustomSpinBox(label="Theta",range_start=0,range_end=360,initial_value=0,step_value=1)
        self.linear_detection_threshold_custom_spin_box = CustomSpinBox(label="Threshold",range_start=0,range_end=100,initial_value=0,step_value=1)
        self.linear_detection_inputs_container_layout.addWidget(self.linear_detection_theta_custom_spin_box)
        self.linear_detection_inputs_container_layout.addWidget(self.linear_detection_threshold_custom_spin_box)


        self.circles_detection_inputs_container = QWidget()
        self.circles_detection_inputs_container_layout = QHBoxLayout(self.circles_detection_inputs_container)
        self.circles_detection_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.circles_detection_inputs_container)
        self.circles_detection_inputs_container.setVisible(False)
        self.circles_detection_threshold_spin_box = CustomSpinBox(label="Threshold",range_start=0,range_end=100,initial_value=0,step_value=1)
        self.circles_detection_min_radius_spin_box = CustomSpinBox(label="Min Radius",range_start=0,range_end=300,initial_value=0,step_value=1)
        self.circles_detection_max_radius_spin_box = CustomSpinBox(label="Max Radius",range_start=0,range_end=300,initial_value=0,step_value=1)
        self.circles_detection_inputs_container_layout.addWidget(self.circles_detection_threshold_spin_box)
        self.circles_detection_inputs_container_layout.addWidget(self.circles_detection_min_radius_spin_box)
        self.circles_detection_inputs_container_layout.addWidget(self.circles_detection_max_radius_spin_box)
        



        self.ellipses_detection_inputs_container = QWidget()
        self.ellipses_detection_inputs_container_layout = QHBoxLayout(self.ellipses_detection_inputs_container)
        self.ellipses_detection_inputs_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.ellipses_detection_inputs_container)
        self.ellipses_detection_inputs_container.setVisible(False)
        self.ellipses_detection_threshold_spin_box = CustomSpinBox(label="Threshold",range_start=0,range_end=100,initial_value=0,step_value=1)
        self.min_major_axis_spin_box = CustomSpinBox(label="Min Major Axis",range_start=0,range_end=100,initial_value=0,step_value=1)
        self.ellipses_detection_inputs_container_layout.addWidget(self.ellipses_detection_threshold_spin_box)
        self.ellipses_detection_inputs_container_layout.addWidget(self.min_major_axis_spin_box)


        self.choosen_color_hex = "red"
        self.choose_color_container = QWidget()
        self.choose_color_container_layout = QVBoxLayout(self.choose_color_container)
        self.choose_color_container_layout.setContentsMargins(0,0,0,0)
        self.inputs_container_layout.addWidget(self.choose_color_container)
        self.choose_color_label = QLabel("Color")
        label_font = QFont("Inter", 9)
        label_font.setWeight(QFont.Medium)
        self.choose_color_label.setFont(label_font)
        self.choose_color_label.setObjectName("choose_color_label")
        self.choose_color_container_layout.addWidget(self.choose_color_label)
        self.choose_color_button = QPushButton("Choose")
        self.choose_color_button.setObjectName("choose_color_button")
        self.choose_color_container_layout.addWidget(self.choose_color_button)
        self.choose_color_button.clicked.connect(self.choose_color)
        self.update_the_color_icon(self.choosen_color_hex)
        self.choose_color_button.setCursor(Qt.PointingHandCursor)

        self.hough_transform_controller = HoughTransformController(self)
        
    def on_detected_objects_type_change(self):
        self.hide_all_inputs()

        selected_detector = self.detected_objects_type_custom_combo_box.current_text()

        match selected_detector:
            case "Lines Detection":
                self.linear_detection_inputs_container.setVisible(True)
            case "Circles Detection":
                self.circles_detection_inputs_container.setVisible(True)
            case "Ellipses Detection":
                self.ellipses_detection_inputs_container.setVisible(True)

    def hide_all_inputs(self):
        self.linear_detection_inputs_container.setVisible(False)
        self.circles_detection_inputs_container.setVisible(False)
        self.ellipses_detection_inputs_container.setVisible(False)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.choosen_color_hex = color.name()
            self.update_the_color_icon(self.choosen_color_hex)

    def update_the_color_icon(self, color_hex):
        pixmap = QPixmap(30, 20)
        pixmap.fill(QColor(color_hex))
        self.choose_color_button.setIcon(QIcon(pixmap))
        self.choose_color_button.setIconSize(QSize(20, 20))

    


