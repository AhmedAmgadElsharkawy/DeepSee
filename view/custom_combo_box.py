from PyQt5.QtWidgets import  QWidget, QComboBox,QLabel,QVBoxLayout
from PyQt5.QtCore import pyqtSignal


class CustomComboBox(QWidget):
    currentIndexChanged = pyqtSignal()

    def __init__(self,label = "",combo_box_items_list = []):
        super().__init__()

        self.centeral_layout = QVBoxLayout(self)
        self.centeral_layout.setContentsMargins(0,0,0,0)

        self.main_widget = QWidget(self)
        self.centeral_layout.addWidget(self.main_widget)
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget_layout.setSpacing(10)

        self.combo_box_label =  QLabel(label)
        self.combo_box = QComboBox()
        self.combo_box.addItems(combo_box_items_list)

        self.main_widget_layout.addWidget(self.combo_box_label)
        self.main_widget_layout.addWidget(self.combo_box)

        self.combo_box.currentIndexChanged.connect(self.on_combobox_change)

    def on_combobox_change(self):
        self.currentIndexChanged.emit()

    def current_text(self):
        return self.combo_box.currentText()
    
    def current_index(self):
        return self.combo_box.currentIndex()

    


