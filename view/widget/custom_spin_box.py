from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox
from PyQt5.QtGui import QFont

class CustomSpinBox(QWidget):
    def __init__(self, label="Label", double_value=False, range_start=0, range_end=100, initial_value=0, decimals=5, step_value=1):
        super().__init__()

        self.central_layout = QVBoxLayout(self)
        self.central_layout.setContentsMargins(0,0,0,0)
        self.main_widget = QWidget()
        self.main_widget_layout = QVBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0,0,0,0)
        self.central_layout.addWidget(self.main_widget)

        self.label = QLabel(label)
        self.label.setFont(QFont("Arial", 10, QFont.Bold))
        # self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #333;")
        self.main_widget_layout.addWidget(self.label)

        if double_value:
            self.spin_box = QDoubleSpinBox()
            self.spin_box.setDecimals(decimals)
            self.spin_box.setSingleStep(step_value)
        else: 
            self.spin_box = QSpinBox()
            step_value = max(1, int(step_value)) 
            initial_value = int(initial_value)
            range_start, range_end = int(range_start), int(range_end)

        self.spin_box.setRange(range_start, range_end)
        self.spin_box.setValue(initial_value)
        self.spin_box.setSingleStep(step_value)
        self.spin_box.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                border: 2px solid gray;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
                background-color: white;
                selection-background-color: #0078D7;
            }
        """)

        self.main_widget_layout.addWidget(self.spin_box)
        # self.spin_box.setButtonSymbols(QDoubleSpinBox.NoButtons)

    def value(self):
        return round(self.spin_box.value(), self.spin_box.decimals()) if isinstance(self.spin_box, QDoubleSpinBox) else self.spin_box.value()

    def setValue(self, value):
        self.spin_box.setValue(value)
