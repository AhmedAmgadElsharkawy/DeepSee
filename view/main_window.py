from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap
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
        self.main_widget.setObjectName("main_widget")
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
            ("Edge Detection", "assets/icons/edge_detection.png"),
            ("Transformations", "assets/icons/transformations.png"),
            ("Hybrid Image", "assets/icons/hybrid_image.png")
        ]

        for name, icon_path in self.list_widget_items:
            item = QListWidgetItem(name)
            icon = self.change_icon_color(icon_path,original_color="black",new_color="#B1B1B1")
            item.setIcon(icon)
            self.list_widget.addItem(item)

        self.nosie_window = NoiseWindow(self)
        self.filters_window = FiltersWindow(self)
        self.thresholding_window = ThresholdingWindow(self)
        self.edge_detection_window = EdgeDetectionsWindow(self)
        self.transformations_window = TransformationsWindow(self)
        self.hybrid_image_widnow = HybridImageWindow(self)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.nosie_window)
        self.stackedWidget.addWidget(self.filters_window)
        self.stackedWidget.addWidget(self.thresholding_window)
        self.stackedWidget.addWidget(self.edge_detection_window)
        self.stackedWidget.addWidget(self.transformations_window)
        self.stackedWidget.addWidget(self.hybrid_image_widnow)

        self.main_widget_layout.addWidget(self.list_widget, 2)
        self.main_widget_layout.addWidget(self.stackedWidget, 10)

        self.list_widget.currentRowChanged.connect(self.on_sidebar_item_select)

        self.list_widget.setCurrentRow(0)
        self.stackedWidget.setCurrentIndex(0)

        self.setStyleSheet("""
            #main_widget {
                background-color: #f5f7fa;
            }      
            #list_widget {
                background-color: white;
                color: #B1B1B1; 
                border: none;
            }

            #list_widget::item {
                padding: 12px;
                margin-top: 4px;
                margin-bottom: 4px;
            }

            #list_widget::item:selected {
                background-color: transparent;
                color: #2D60FF; 
                font-weight: bold;
                border-left: 4px solid #2D60FF;
            }

            #list_widget::item:hover:!selected {
                background-color: #F8F8F8; 
            }
                           
            QLabel#header_label {
                background-color: #f5f7fa;
            }
                           
            QPushButton#apply_button {
                font-size: 18px;
                font-weight: bold;
                padding: 8px 25px;
                border: 2px solid #888888;
                border-radius: 8px;
                background-color: #E0E0E0;
                color: #333333;
            }

            QPushButton#apply_button:hover {
                background-color: #D0D0D0;
                border-color: #777777;
            }

            QPushButton#apply_button:pressed {
                background-color: #B0B0B0;
                border-color: #666666;
            }

            QPushButton#apply_button:disabled {
                background-color: #C0C0C0;
                border-color: #A0A0A0;
                color: #666666;
            }
                           
            QSpinBox#spin_box, QDoubleSpinBox#combo_box {
                border: 2px solid gray;
                border-radius: 5px;
                padding: 3px;
                font-size: 12px;
                background-color: white;
                selection-background-color: #0078D7;
            }
            QLabel#spin_box_label , QLabel#combo_box_label{
                color: #333;
            }
                           
            QComboBox#combo_box {
                border: 2px solid gray;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
                background-color: white;
                selection-background-color: #0078D7;
            }
        """)


    def change_icon_color(self,icon_path,original_color,new_color):
        # note the image must have only one solid color to mask it in a right way
        pixmap = QPixmap(icon_path)
        mask = pixmap.createMaskFromColor(QColor(original_color), Qt.MaskOutColor)
        pixmap.fill((QColor(new_color)))
        pixmap.setMask(mask)
        return QIcon(pixmap)
    
    def on_sidebar_item_select(self,index):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            icon_path = self.list_widget_items[i][1]
            
            if i == index:
                icon = self.change_icon_color(icon_path,original_color="black",new_color= "#2D60FF")
            else:
                icon = self.change_icon_color(icon_path,original_color="black", new_color="#B1B1B1")
            
            item.setIcon(icon)

        self.stackedWidget.setCurrentIndex(index)


