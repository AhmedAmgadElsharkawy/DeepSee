from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt

from view.window.noise_window import NoiseWindow
from view.window.filters_window import FiltersWindow
from view.window.thresholding_window import ThresholdingWindow
from view.window.edge_detection_window import EdgeDetectionsWindow
from view.window.transformations_window import TransformationsWindow
from view.window.hybrid_image_window import HybridImageWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('DeepSee')

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_widget_layout = QHBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget_layout.setSpacing(10)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("list_widget")
        self.list_widget.setFont(QFont("Arial", 10,QFont.Bold))
        self.list_widget.setFocusPolicy(Qt.NoFocus)


        self.list_widget_items = [
            ("Noise", "assets/icons/noise.png"),
            ("Filters", "assets/icons/filter.png"),
            ("Thresholding", "assets/icons/thresholding.png"),
            ("Edge Detection", "assets/icons/edge.png"),
            ("Transformations", "assets/icons/transformations.png"),
            ("Hybrid Image", "assets/icons/hybrid_image.png")
        ]

        for name, icon_path in self.list_widget_items:
            item = QListWidgetItem(name)
            item.setIcon(QIcon(icon_path))
            self.list_widget.addItem(item)

        self.nosie_window = NoiseWindow()
        self.filters_window = FiltersWindow()
        self.thresholding_window = ThresholdingWindow()
        self.edge_detection_window = EdgeDetectionsWindow()
        self.transformations_window = TransformationsWindow()
        self.hybrid_image_widnow = HybridImageWindow()

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.nosie_window)
        self.stackedWidget.addWidget(self.filters_window)
        self.stackedWidget.addWidget(self.thresholding_window)
        self.stackedWidget.addWidget(self.edge_detection_window)
        self.stackedWidget.addWidget(self.transformations_window)
        self.stackedWidget.addWidget(self.hybrid_image_widnow)

        self.main_widget_layout.addWidget(self.list_widget, 2)
        self.main_widget_layout.addWidget(self.stackedWidget, 10)

        self.list_widget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)

        self.list_widget.setCurrentRow(0)
        self.stackedWidget.setCurrentIndex(0)

        self.setStyleSheet("""
            #list_widget {
                background-color: #E5E5E5;
                color: #888;
                border-right: 2px solid #B0B0B0;
            }
            #list_widget::item {
                padding: 12px;
                border-bottom: 1px solid #D0D0D0;
            }
            #list_widget::item:selected {
                background-color: #C0C0C0;
                color: #444;
                font-weight: bold;
            }
            #list_widget::item:hover {
                background-color: #D0D0D0;
            }           
            """)