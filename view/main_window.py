from PyQt5.QtWidgets import QMainWindow,QSlider,QWidget,QHBoxLayout,QVBoxLayout,QLabel,QPushButton,QComboBox,QSpinBox,QDoubleSpinBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('DeepSee')

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setSpacing(10)


        
        
