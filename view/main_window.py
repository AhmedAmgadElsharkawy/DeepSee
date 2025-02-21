from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem
from PyQt5.QtGui import  QIcon

from view.noise_window import NoiseWindow
from view.filters_window import FiltersWindow
from view.thresholding_window import ThresholdingWindow
from view.edge_detection_window import EdgeDetectionsWindow
from view.transformations_window import TransformationsWindow
from view.hybrid_image_window import HybridImageWindow


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
            ("Noise", "assets/icons/noise.png"),
            ("Filters", "assets/icons/filter.png"),
            ("Thresholding","assets/icons/thresholding.png"),
            ("Edge Detection", "assets/icons/edge.png"),
            ("Transformations", "assets/icons/transformations.png"),
            ("Hybrid Image", "assets/icons/hybrid_image.png")
        ]

        for name, icon_path in self.list_widget_items:
            item = QListWidgetItem(name)
            item.setIcon(QIcon(icon_path))
            self.listWidget.addItem(item)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(NoiseWindow())
        self.stackedWidget.addWidget(FiltersWindow())
        self.stackedWidget.addWidget(ThresholdingWindow())
        self.stackedWidget.addWidget(EdgeDetectionsWindow())
        self.stackedWidget.addWidget(TransformationsWindow())
        self.stackedWidget.addWidget(HybridImageWindow())

        self.main_widget_layout.addWidget(self.listWidget, 1)
        self.main_widget_layout.addWidget(self.stackedWidget, 6)

        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)


