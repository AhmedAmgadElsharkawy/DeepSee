from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from view.widget.interactive_image_viewer import InteractiveImageViewer
from view.widget.image_viewer import ImageViewer


class BasicStackedWindow(QWidget):
    def __init__(self, main_window, header_text=""):
        super().__init__()
        self.main_window = main_window

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(10, 10, 10, 10)

        self.main_widget = QWidget(self)
        self.main_widget.setObjectName("main_widget")
        self.central_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setSpacing(15)

        self.header_label = QLabel(header_text)
        self.header_label.setObjectName("header_label")  # Set object name once
        font = QFont("Arial", 24, QFont.Bold)
        self.header_label.setFont(font)
        self.header_label.setAlignment(Qt.AlignCenter)

        self.image_viewers_container = QWidget()
        self.image_viewers_container_layout = QHBoxLayout(self.image_viewers_container)
        self.image_viewers_container_layout.setContentsMargins(0, 0, 0, 0)
        self.image_viewers_container_layout.setSpacing(10)

        self.input_image_viewer = InteractiveImageViewer()
        self.output_image_viewer = ImageViewer()

        self.image_viewers_container_layout.addWidget(self.input_image_viewer)
        self.image_viewers_container_layout.addWidget(self.output_image_viewer)

        self.controls_container = QWidget()
        self.controls_container.setObjectName("controls_container")
        self.controls_container_layout = QHBoxLayout(self.controls_container)
        self.controls_container_layout.setDirection(QHBoxLayout.RightToLeft)
        self.controls_container_layout.setContentsMargins(0, 0, 0, 0)

        self.buttons_container = QWidget()
        self.buttons_container_layout = QHBoxLayout(self.buttons_container)
        self.buttons_container_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_container_layout.addWidget(self.buttons_container)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("apply_button")
        self.apply_button.setCursor(Qt.PointingHandCursor)

        self.buttons_container_layout.addWidget(self.apply_button)

        self.controls_container_layout.addStretch()

        self.main_widget_layout.addWidget(self.header_label)
        self.main_widget_layout.addWidget(self.controls_container)
        self.main_widget_layout.addWidget(self.image_viewers_container)

        self.inputs_container = QWidget()
        self.inputs_container_layout = QHBoxLayout(self.inputs_container)
        self.inputs_container_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_container_layout.addWidget(self.inputs_container)

