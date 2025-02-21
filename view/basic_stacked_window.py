from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from view.interactive_image_viewer import InteractiveImageViewer
from view.image_viewer import ImageViewer


class BasicStackedWindow(QWidget):
    def __init__(self, header_text=""):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(10, 10, 10, 10)

        self.main_widget = QWidget(self)
        self.central_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(10, 10, 10, 10)
        self.main_widget_layout.setSpacing(15)

        self.header_label = QLabel(header_text)
        font = QFont("Arial", 24, QFont.Bold)
        self.header_label.setFont(font)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("""
            color: #333;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        """)

        self.image_viewers_container = QWidget()
        self.image_viewers_container_layout = QHBoxLayout(self.image_viewers_container)
        self.image_viewers_container_layout.setContentsMargins(0, 0, 0, 0)
        self.image_viewers_container_layout.setSpacing(10)

        self.input_image_viewer = InteractiveImageViewer()
        self.output_image_viewer = ImageViewer()

        self.image_viewers_container_layout.addWidget(self.input_image_viewer)
        self.image_viewers_container_layout.addWidget(self.output_image_viewer)

        self.controls_container = QWidget()
        self.controls_container_layout = QHBoxLayout(self.controls_container)
        self.controls_container_layout.setDirection(QHBoxLayout.RightToLeft)
        self.controls_container_layout.setContentsMargins(0, 0, 0, 0)

        self.buttons_container = QWidget()
        self.buttons_container_layout = QHBoxLayout(self.buttons_container)
        self.buttons_container_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_container_layout.addWidget(self.buttons_container)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setCursor(Qt.PointingHandCursor)
        self.apply_button.setStyleSheet("""
        QPushButton {
            font-size: 18px;
            font-weight: bold;
            padding: 8px 25px;
            border: 2px solid #888888;
            border-radius: 8px;
            background-color: #E0E0E0;
            color: #333333;
        }
        
        QPushButton:hover {
            background-color: #D0D0D0;
            border-color: #777777;
        }

        QPushButton:pressed {
            background-color: #B0B0B0;
            border-color: #666666;
        }

        QPushButton:disabled {
            background-color: #C0C0C0;
            border-color: #A0A0A0;
            color: #666666;
        }
    """)


        self.buttons_container_layout.addWidget(self.apply_button)

        self.controls_container_layout.addStretch()

        self.main_widget_layout.addWidget(self.header_label)
        self.main_widget_layout.addWidget(self.controls_container)
        self.main_widget_layout.addWidget(self.image_viewers_container)

        self.inputs_container = QWidget()
        self.inputs_container_layout = QHBoxLayout(self.inputs_container)
        self.inputs_container_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_container_layout.addWidget(self.inputs_container)



