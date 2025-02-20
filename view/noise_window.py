from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QListWidget, QStackedWidget, QVBoxLayout, QLabel, QHBoxLayout, QListWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

from view.interactive_image_viewer import InteractiveImageViewer
from view.image_viewer import ImageViewer

class NoiseWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(0,0,0,0)
        self.main_widget = QWidget(self)
        self.central_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.setSpacing(10)

        self.header_widget = QLabel("Noise")
        self.main_widget_layout.addWidget(self.header_widget)

        self.image_viewer_container = QWidget()
        self.image_viewer_container_layout = QHBoxLayout(self.image_viewer_container)
        self.main_widget_layout.addWidget(self.image_viewer_container)
        self.image_viewer_container_layout.setContentsMargins(0,0,0,0)

        self.input_image_viewer = InteractiveImageViewer()
        self.output_image_viewer = ImageViewer()

        self.image_viewer_container_layout.addWidget(self.input_image_viewer)
        self.image_viewer_container_layout.addWidget(self.output_image_viewer)

        


        

    


