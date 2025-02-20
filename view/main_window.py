from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QStackedWidget, QVBoxLayout, QLabel, QHBoxLayout, QListWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

from view.noise_window import NoiseWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('DeepSee')

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_widget_layout = QHBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget_layout.setSpacing(10)

        self.listWidget = QListWidget()

        self.list_widget_items = [
            ("Noise", "icons/noise.png"),
            ("Filters", "icons/filter.png"),
            ("Edge Detection", "icons/edge.png"),
            ("Hybrid Image", "icons/hybrid.png")
        ]

        for name, icon_path in self.list_widget_items:
            item = QListWidgetItem(name)  # ✅ Corrected
            item.setIcon(QIcon(icon_path))
            self.listWidget.addItem(item)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(NoiseWindow())
        self.stackedWidget.addWidget(self.create_page("Filter Settings"))
        self.stackedWidget.addWidget(self.create_page("Edge Detection Tools"))
        self.stackedWidget.addWidget(self.create_page("Hybrid Image Processor"))

        self.main_widget_layout.addWidget(self.listWidget, 1)
        self.main_widget_layout.addWidget(self.stackedWidget, 3)

        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)

    def create_page(self, text):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(text)
        label.setFont(QFont("Arial", 14))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return page


