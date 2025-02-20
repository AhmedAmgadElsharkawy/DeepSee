from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QListWidget, QStackedWidget, QVBoxLayout, QLabel, QHBoxLayout, QListWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon


class FiltersWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget_layout.setSpacing(10)

    


