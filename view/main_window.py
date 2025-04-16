from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem,QVBoxLayout,QLabel,QPushButton
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap
from PyQt5.QtCore import Qt,pyqtSignal

from view.window.noise_window import NoiseWindow
from view.window.filters_window import FiltersWindow
from view.window.thresholding_window import ThresholdingWindow
from view.window.edge_detection_window import EdgeDetectionsWindow
from view.window.transformations_window import TransformationsWindow
from view.window.hybrid_image_window import HybridImageWindow
from view.window.hough_transform_window import HoughTransformWindow
from view.window.active_contours_window import ActiveContoursWindow
from view.window.image_matching_window import ImageMatchingWindow
from view.window.corner_detection_window import CornerDetectionWindow
from view.window.sift_descriptors_window import SiftDescriptorsWindow
from view.window.segmentation_window import SegmentationWindow
from view.window.face_detection_and_recognition_window import FaceDetectionAndRecognitionWindow

class MainWindow(QMainWindow):
    __instance = None
    mode_toggle_signal = pyqtSignal(bool)

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            return super(MainWindow, cls).__new__(cls)
        return cls.__instance     

    def __init__(self):
        if MainWindow.__instance != None:
            return
        
        super().__init__() 
        MainWindow.__instance = self

        self.is_dark_mode = False
        self.setWindowIcon(QIcon("assets/icons/deepsee.png"))

        self.setWindowTitle('DeepSee')

        self.main_widget = QWidget(self)
        self.main_widget.setObjectName("main_widget")
        self.setCentralWidget(self.main_widget)
        self.main_widget_layout = QHBoxLayout(self.main_widget)
        self.main_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget_layout.setSpacing(10)

        self.side_bar_container = QWidget()
        self.side_bar_container_layout = QVBoxLayout(self.side_bar_container)
        self.side_bar_container.setObjectName("side_bar")
        self.side_bar_container_layout.setContentsMargins(0,0,0,0)

        self.list_widget = QListWidget()
        self.side_bar_container_layout.addWidget(self.list_widget)
        self.list_widget.setObjectName("list_widget")
        self.list_widget.setFocusPolicy(Qt.NoFocus)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


        self.list_widget_items = [
            ("Noise", "assets/icons/noise.png"),
            ("Filters", "assets/icons/filter.png"),
            ("Thresholding", "assets/icons/thresholding.png"),
            ("Edge Detection", "assets/icons/edge_detection.png"),
            ("Transformations", "assets/icons/transformations.png"),
            ("Hybrid Image", "assets/icons/hybrid_image.png"),
            ("Hough Transform","assets/icons/hough_transform.png"),
            ("Active Contours","assets/icons/active_contours.png"),
            ("Image Matching","assets/icons/image_matching.png"),
            ("Corner Detection","assets/icons/corner_detection.png"),
            ("SIFT Descriptors","assets/icons/sift_descriptors.png"),
            ("Segmentation","assets/icons/segmentation.png"),
            ("Face Detection & Recognition","assets/icons/face_detection_and_recognition.png")
        ]

        font = QFont("Inter",11)
        font.setWeight(QFont.DemiBold)  

        for name, icon_path in self.list_widget_items:
            item = QListWidgetItem(name)
            icon = self.change_icon_color(icon_path, original_color="black", new_color="#B1B1B1")
            item.setIcon(icon)
            item.setFont(font)
            self.list_widget.addItem(item)

        self.nosie_window = NoiseWindow(self)
        self.filters_window = FiltersWindow(self)
        self.thresholding_window = ThresholdingWindow(self)
        self.edge_detection_window = EdgeDetectionsWindow(self)
        self.transformations_window = TransformationsWindow(self)
        self.hybrid_image_window = HybridImageWindow(self)
        self.hough_transform_window = HoughTransformWindow(self)
        self.active_contours_window = ActiveContoursWindow(self)
        self.image_matching_window = ImageMatchingWindow(self)
        self.corner_detection_window = CornerDetectionWindow(self)
        self.sift_descriptors_window = SiftDescriptorsWindow(self)
        self.segmentation_window = SegmentationWindow(self)
        self.Face_detection_and_recognition_window = FaceDetectionAndRecognitionWindow(self)
        

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.nosie_window)
        self.stackedWidget.addWidget(self.filters_window)
        self.stackedWidget.addWidget(self.thresholding_window)
        self.stackedWidget.addWidget(self.edge_detection_window)
        self.stackedWidget.addWidget(self.transformations_window)
        self.stackedWidget.addWidget(self.hybrid_image_window)
        self.stackedWidget.addWidget(self.hough_transform_window)
        self.stackedWidget.addWidget(self.active_contours_window)
        self.stackedWidget.addWidget(self.image_matching_window)
        self.stackedWidget.addWidget(self.corner_detection_window)
        self.stackedWidget.addWidget(self.sift_descriptors_window)
        self.stackedWidget.addWidget(self.segmentation_window)
        self.stackedWidget.addWidget(self.Face_detection_and_recognition_window)
        
        

        self.side_bar_container.setFixedWidth(330)
        self.main_widget_layout.addWidget(self.side_bar_container)
        self.main_widget_layout.addWidget(self.stackedWidget)


        self.dark_mode_toggle_button_container = QWidget()
        self.dark_mode_toggle_button_container.setObjectName("dark_mode_toggle_button_container")
        self.dark_mode_toggle_button_container_layout = QHBoxLayout(self.dark_mode_toggle_button_container)
        self.dark_mode_toggle_button_container_layout.setContentsMargins(20,10,20,10)
        self.side_bar_container_layout.addWidget(self.dark_mode_toggle_button_container)
        self.mode_toggle_button_label = QLabel("Change The Theme")
        mode_toggle_button_label_font = QFont("Inter",11)
        self.mode_toggle_button_label.setFont(mode_toggle_button_label_font)
        self.mode_toggle_button_label.setObjectName("mode_toggle_button_label")
        self.dark_mode_toggle_button_container_layout.addWidget(self.mode_toggle_button_label)
        self.mode_toggle_button = QPushButton()
        self.mode_toggle_button.setFixedSize(40,40)
        self.mode_toggle_button.setObjectName("mode_toggle_button")
        self.dark_mode_toggle_button_container_layout.addWidget(self.mode_toggle_button)
        self.mode_toggle_button.setIcon(QIcon("assets/icons/sun.png"))
        self.mode_toggle_button.clicked.connect(self.toggle_theme)
        self.mode_toggle_button.setCursor(Qt.PointingHandCursor)

        self.list_widget.currentRowChanged.connect(self.on_sidebar_item_select)

        self.list_widget.setCurrentRow(0)
        self.stackedWidget.setCurrentIndex(0)

        self.setStyleSheet(self.light_mode_stylesheet())




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
                if self.is_dark_mode:
                    icon = self.change_icon_color(icon_path,original_color="black",new_color= "#FFFFFF")
                else:
                    icon = self.change_icon_color(icon_path,original_color="black",new_color= "#2D60FF")
            else:
                icon = self.change_icon_color(icon_path,original_color="black", new_color="#B1B1B1")
            
            item.setIcon(icon)

        self.stackedWidget.setCurrentIndex(index)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode

        if self.is_dark_mode:
            self.setStyleSheet(self.dark_mode_stylesheet())
            self.mode_toggle_button.setIcon(QIcon("assets/icons/moon.png"))
        else:
            self.setStyleSheet(self.light_mode_stylesheet())
            self.mode_toggle_button.setIcon(QIcon("assets/icons/sun.png"))

        self.mode_toggle_signal.emit(self.is_dark_mode)
        self.on_sidebar_item_select(self.list_widget.currentRow())

    def light_mode_stylesheet(self):
        return """
            #main_widget {
                background-color: #f5f7fa;
            }      
            #list_widget {
                background-color: white;
                color: #B1B1B1; 
                border: none;
            }
            #side_bar{
                 background-color: white;
                color: #B1B1B1; 
                border: none;          
            }

            #list_widget::item {
                padding: 10px 15px;
                margin-top: 4px;
                margin-bottom: 4px;
                background-color: transparent; 
            }

            #list_widget::item:selected {
                color: #2D60FF; 
                font-weight: bold;
                border-left: 4px solid #2D60FF;
            }

            #list_widget::item:hover:!selected {
                background-color: #F8F8F8; 
            }
                           
            QLabel#header_label {
                background-color: #f5f7fa;
                color: #343C6A;
            }

            QLabel#mode_toggle_button_label{
                color: #343C6A;
            }

            QPushButton#mode_toggle_button{
                padding: 8px 8px;
                border: 1px solid #B1B1B1;
                border-radius: 8px;
                background-color: #FFFFFF;
            }

            QPushButton#mode_toggle_button:hover{
                padding: 8px 8px;
                border: 2px solid #B1B1B1;
                border-radius: 8px;
                background-color: #B1B1B1;
            }
                           
            QPushButton#apply_button {
                font-size: 18px;
                font-weight: bold;
                padding: 8px 25px;
                border-radius: 8px;
                background-color: #2D60FF;
                color: #FFFFFF;
            }

            QPushButton#apply_button:hover {
                background-color: #FFFFFF;
                border: 1px solid #2D60FF;
                color:#2D60FF;
            }

            QPushButton#apply_button:pressed {
                background-color: #B0B0B0;
                border-color: #666666;
            }

            QPushButton#apply_button:disabled {
                background-color: #C0C0C0;
                border-color: #A0A0A0;
                color: #B1B1B1;
            }
                           
            #spin_box {
                border: 1px solid #343C6A;
                padding: 3px;
                font-size: 12px;
                background-color: #f5f7fa;
                selection-background-color: #0078D7;
            }
                           

            #spin_box::up-button {
                subcontrol-position: top right;
                width: 16px; /* Adjust width */
                background: #f5f7fa; /* Background color */

            }

            #spin_box::down-button {
                subcontrol-position: bottom right;
                width: 16px;
                background: #f5f7fa;

            }

            #spin_box::up-arrow {
                image: url(assets/icons/up_arrow.png); /* Replace with your custom up arrow */
                width: 10px;
                height: 10px;
            }

            #spin_box::down-arrow {
                image: url(assets/icons/down_arrow.png); /* Replace with your custom down arrow */
                width: 10px;
                height: 10px;
            }

            #spin_box::up-button:pressed, #spin_box::down-button:pressed {
                background: #A1A1A1;
            }
                           
            QLabel#spin_box_label , QLabel#combo_box_label, #choose_color_label{
                color: #343C6A;
            }
                           
            QComboBox#combo_box {
                border: 1px solid #343C6A;
                padding: 5px;
                color: #343C6A;
                background-color: #f5f7fa;
                selection-background-color: #f5f7fa;
            }
            QComboBox#combo_box QAbstractItemView {
                background-color: #f5f7fa;
                color: #343C6A;
                selection-background-color: #F8F8F8;
                selection-color: #343C6A;
            }
                           
            QComboBox#combo_box::drop-down {
                background-color: #f5f7fa; /* Change this to your desired color */
                width: 20px; /* Adjust the width */
            }

            QComboBox#combo_box::down-arrow {
                image: url(assets/icons/down_arrow.png); /* Set a custom arrow icon if needed */
                width: 10px;
                height: 10px;
            }
         
            ImageViewer QLabel#temp_label{
                    color: #A0A0A0;
                    background: transparent;     
                }
            ImageViewer QPushButton#save_image_button{
                    border: none;
                    background-color: transparent;   
                }
            ImageViewer QPushButton#save_image_button:hover{
                    background-color: none;
                }
            
            #choose_color_button{
                border: 1px solid #343C6A;
                padding: 3px 10px;
                margin:0px;
                color: #343C6A;
                background-color: #f5f7fa;
                border-radius:4px;      
            }
                           
        #choose_color_button:hover {
            background-color: #2D60FF; 
            color: #FFFFFF;
            border-color: none; 
        }              

        QScrollBar:vertical {
            background: #FFFFFF;
            width: 8px;
            margin: 0px;
            border-radius: 4px;
        }

        QScrollBar::handle:vertical {
            background: #2D60FF;
            min-height: 20px;
            border-radius: 4px;
        }

        QScrollBar::handle:vertical:hover {
            background: #2D60FF;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: none;
        }

        #active_contours_detector_perimeter_label,#active_contours_detector_area_label,#active_contours_detector_chaincode_label,#active_contours_detector_chaincode{
            color: #343C6A;
            background-color: #f5f7fa;
        }

        #active_contours_detector_area,#active_contours_detector_perimeter{
            color: #343C6A;
            background-color: #f5f7fa;
            border: none;
        }

        """

    def dark_mode_stylesheet(self):
        return """
            #main_widget {
                background-color: #1B2431;
            }      
            #list_widget {
                background-color: #273142;
                color: #B1B1B1; 
                border: none;
            }
            #side_bar{
                background-color: #273142;
                color: #B1B1B1; 
                border: none;          
            }

            #list_widget::item {
                padding: 10px 15px;
                margin-top: 4px;
                margin-bottom: 4px;
                background-color: transparent; 
            }

            #list_widget::item:selected {
                color: #FFFFFF; 
                font-weight: bold;
                background-color: #4B5668;
                border-left: 4px solid #2D60FF;
            }

            #list_widget::item:hover:!selected {
                background-color: #4B5668; 
            }
                           
            QLabel#header_label {
                background-color: #1B2431;
                color: #FFFFFF;
            }

            QLabel#mode_toggle_button_label{
                color: #FFFFFF;
            }

            QPushButton#mode_toggle_button{
                padding: 8px 8px;
                border-radius: 8px;
                background-color: #2D60FF;
            }

            QPushButton#mode_toggle_button:hover{
                padding: 8px 8px;
                border: 2px solid #2D60FF;
                border-radius: 8px;
                background-color: #273142;
            }
                           
            QPushButton#apply_button {
                font-size: 18px;
                font-weight: bold;
                padding: 8px 25px;
                border-radius: 8px;
                background-color: #2D60FF;
                color: #FFFFFF;
            }

            QPushButton#apply_button:hover {
                background-color: #1B2431;
                border: 1px solid #2D60FF;
                color:#FFFFFF;
            }

            QPushButton#apply_button:pressed {
                background-color: #B0B0B0;
                border-color: #666666;
            }

            QPushButton#apply_button:disabled {
                background-color: #C0C0C0;
                border-color: #A0A0A0;
                color: #B1B1B1;
            }
                           
            #spin_box {
                border: 1px solid #343C6A;
                padding: 3px;
                font-size: 12px;
                color: #FFFFFF;
                background-color: #273142;
                selection-background-color: #4B5668;
            }
                           

            #spin_box::up-button {
                subcontrol-position: top right;
                width: 16px; /* Adjust width */
                background: #273142;
            }

            #spin_box::down-button {
                subcontrol-position: bottom right;
                width: 16px;
                background: #273142;

            }

            #spin_box::up-arrow {
                image: url(assets/icons/white_up_arrow.png); /* Replace with your custom up arrow */
                width: 10px;
                height: 10px;
            }

            #spin_box::down-arrow {
                image: url(assets/icons/white_down_arrow.png); /* Replace with your custom down arrow */
                width: 10px;
                height: 10px;
            }

            #spin_box::up-button:pressed, #spin_box::down-button:pressed {
                background: #B0B0B0;
            }
                           
            QLabel#spin_box_label , QLabel#combo_box_label, #choose_color_label{
                color: #FFFFFF;
            }
                           
            QComboBox#combo_box {
                border: 1px solid #343C6A;
                padding: 5px;
                color: #FFFFFF;
                background-color: #273142;
                selection-background-color: #FFFFFF;
            }
            QComboBox#combo_box QAbstractItemView {
                background-color: #273142;
                color: #FFFFFF;
                selection-background-color: #4B5668;
                selection-color: #FFFFFF;
            }
                           
            QComboBox#combo_box::drop-down {
                background-color: #273142; 
                width: 20px; 
            }

            QComboBox#combo_box::down-arrow {
                image: url(assets/icons/white_down_arrow.png); 
                width: 10px;
                height: 10px;
            }
         
            ImageViewer QLabel#temp_label{
                    color: #A0A0A0;
                    background: transparent;     
                }
            ImageViewer QPushButton#save_image_button{
                    border: none;
                    background-color: transparent;   
                }
            ImageViewer QPushButton#save_image_button:hover{
                    background-color: none;
                }
            
            #choose_color_button{
                border: 1px solid #273142;
                padding: 3px 10px;
                margin:0px;
                color: #FFFFFF;
                background-color: #273142;
                border-radius:4px;      
            }
                           
            #choose_color_button:hover {
                background-color: #2D60FF; 
                color: #FFFFFF;
                border-color: #2D60FF; 
            }              

            QScrollBar:vertical {
                background: #273142;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical {
                background: #2D60FF;
                min-height: 20px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical:hover {
                background: #2D60FF;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }

            #active_contours_detector_perimeter_label,#active_contours_detector_area_label,#active_contours_detector_chaincode_label,#active_contours_detector_chaincode{
                color: #FFFFFF;
                background-color: #1B2431;
            }

            #active_contours_detector_area,#active_contours_detector_perimeter{
                color: #FFFFFF;
                background-color: #1B2431;
                border: none;
            }
        """



