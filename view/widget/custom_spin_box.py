from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal

class CustomSpinBox(QWidget):

    # Expose a new signal that mirrors the internal spin box signal
    valueChanged = pyqtSignal(float)
    def __init__(self, label="Label", double_value=False, range_start=0, range_end=100, initial_value=0, decimals=5, step_value=1):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(0,0,0,0)
        self.main_widget = QWidget()
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0,0,10,0)
        self.main_widget_layout.setSpacing(5)
        self.central_layout.addWidget(self.main_widget)

        self.spin_box_label = QLabel(label)
        font = QFont("Inter", 9)
        font.setWeight(QFont.Medium)
        self.spin_box_label.setFont(font)
        self.spin_box_label.setObjectName("spin_box_label")
        # self.label.setAlignment(Qt.AlignCenter)
        self.main_widget_layout.addWidget(self.spin_box_label)

        if double_value:
            self.spin_box = QDoubleSpinBox()
            self.spin_box.setDecimals(decimals)
            self.spin_box.setSingleStep(step_value)
        else: 
            self.spin_box = QSpinBox()
            step_value = max(1, int(step_value)) 
            initial_value = int(initial_value)
            range_start, range_end = int(range_start), int(range_end)

        self.spin_box.setObjectName("spin_box")

        self.spin_box.setRange(range_start, range_end)
        self.spin_box.setValue(initial_value)
        self.spin_box.setSingleStep(step_value)
        

        self.main_widget_layout.addWidget(self.spin_box)
        # self.spin_box.setButtonSymbols(QDoubleSpinBox.NoButtons)
        # Connect internal spinbox valueChanged to the external signal
        self.spin_box.valueChanged.connect(self.emit_value_changed)


    def value(self):
        return round(self.spin_box.value(), self.spin_box.decimals()) if isinstance(self.spin_box, QDoubleSpinBox) else self.spin_box.value()

    def setValue(self, value):
        self.spin_box.setValue(value)

    def emit_value_changed(self, value):
        """
        Emit valueChanged from CustomSpinBox when internal spinbox changes.
        Always emit float to be consistent.
        """
        self.valueChanged.emit(float(value))

    def setMaximum(self, max_value):
        if isinstance(self.spin_box, QSpinBox):
            max_value = int(max_value)
        self.spin_box.setMaximum(max_value)

    def setMinimum(self, min_value):
        if isinstance(self.spin_box, QSpinBox):
            min_value = int(min_value)
        self.spin_box.setMinimum(min_value)

