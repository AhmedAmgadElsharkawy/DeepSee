from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout,QPushButton

from view.interactive_image_viewer import InteractiveImageViewer
from view.image_viewer import ImageViewer


class BasicStackedWindow(QWidget):
    def __init__(self,header_text = ""):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(0,0,0,0)
        self.main_widget = QWidget(self)
        self.central_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.setSpacing(10)

        self.header_widget = QLabel(header_text)

        self.image_viewers_container = QWidget()
        self.image_viewers_container_layout = QHBoxLayout(self.image_viewers_container)
        self.image_viewers_container_layout.setContentsMargins(0,0,0,0)

        self.input_image_viewer = InteractiveImageViewer()
        self.output_image_viewer = ImageViewer()

        self.image_viewers_container_layout.addWidget(self.input_image_viewer)
        self.image_viewers_container_layout.addWidget(self.output_image_viewer)

        self.controls_container = QWidget()
        self.controls_container_layout = QHBoxLayout(self.controls_container)
        self.controls_container_layout.setDirection(QHBoxLayout.RightToLeft)
        self.controls_container_layout.setContentsMargins(0,0,0,0)

        self.buttons_container = QWidget()
        self.buttons_container_layout = QHBoxLayout(self.buttons_container)
        self.buttons_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.buttons_container)
        
        self.apply_button = QPushButton("Apply")
        # self.save_button = QPushButton("Save")
        self.buttons_container_layout.addWidget(self.apply_button)
        # self.buttons_container_layout.addWidget(self.save_button)

        self.controls_container_layout.addStretch()

        self.main_widget_layout.addWidget(self.header_widget)
        self.main_widget_layout.addWidget(self.controls_container)
        self.main_widget_layout.addWidget(self.image_viewers_container)

        self.inputs_container = QWidget()
        self.inputs_container_layout = QHBoxLayout(self.inputs_container)
        self.inputs_container_layout.setContentsMargins(0,0,0,0)
        self.controls_container_layout.addWidget(self.inputs_container)




        


        

    


