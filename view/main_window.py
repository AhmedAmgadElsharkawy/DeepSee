from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem,QVBoxLayout,QLabel,QPushButton,QToolButton
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap
from PyQt5.QtCore import Qt,pyqtSignal, QEvent, QSize, QPoint

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



class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initial_pos = None
        central_layout = QHBoxLayout(self)
        central_layout.setContentsMargins(0,0,0,0)
        title_bar = QWidget()
        self.setFixedHeight(35)
        title_bar.setObjectName("title_bar")
        central_layout.addWidget(title_bar)
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(10, 0,0, 0)
        
        self.title = QLabel (self)
        font = QFont("Inter",10)
        font.setWeight(QFont.DemiBold)  
        self.title.setFont(font)
        self.title.setObjectName("title_bar_label")
        self.title.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.title.setText("DeepSee")
        title_bar_layout.addWidget(self.title)

        self.icon = QLabel()
        self.icon.setFixedSize(24, 24) 
        self.icon.setScaledContents(True)
        self.icon.setPixmap(QPixmap("assets/icons/deepsee.png")) 

        title_bar_layout.addWidget(self.icon)
        title_bar_layout.addWidget(self.title)

        self.min_button = QToolButton(self)
        min_icon = QIcon()
        self.min_button.setIcon(min_icon)
        self.min_button.clicked.connect(self.window().showMinimized)

        self.max_button = QToolButton(self)
        max_icon = QIcon()
        self.max_button.setIcon(max_icon)
        self.max_button.clicked.connect(self.window().showMaximized)

        self.close_button = QToolButton(self)
        close_icon = QIcon()
        self.close_button.setIcon(close_icon)
        self.close_button.clicked.connect(self.window().close)

        self.normal_button = QToolButton(self)
        normal_icon = QIcon()
        self.normal_button.setIcon(normal_icon)
        self.normal_button.clicked.connect(self.window().showNormal)
        self.normal_button.setVisible(False)

        buttons = [
            self.min_button,
            self.normal_button,
            self.max_button,
            self.close_button,
        ]

        self.buttons_widget = QWidget()
        self.buttons_widget_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_widget_layout.setContentsMargins(10,0,10,0)
        self.buttons_widget_layout.setSpacing(10)
        title_bar_layout.addWidget(self.buttons_widget)
        self.buttons_widget.setFixedWidth(120)
        
        for button in buttons:
            button.setFocusPolicy(Qt.NoFocus)
            button.setFixedSize(QSize(16, 16))
            self.buttons_widget_layout.addWidget(button)


        self.close_button.setStyleSheet("""
            QToolButton {
                background: #FF5F56;
                border: none;
                border-radius: 8px;
            }
            QToolButton:hover {
                background: #E0443E;
            }
        """)

        # Minimize button - Yellow
        self.min_button.setStyleSheet("""
            QToolButton {
                background: #FFBD2E;
                border: none;
                border-radius: 8px;
            }
            QToolButton:hover {
                background: #FFA500;
            }
        """)

        # Maximize button - Green
        self.max_button.setStyleSheet("""
            QToolButton {
                background: #27C93F;
                border: none;
                border-radius: 8px;
            }
            QToolButton:hover {
                background: #1AAB29;
            }
        """)

        self.normal_button.setStyleSheet("""
            QToolButton {
                background: #27C93F;
                border: none;
                border-radius: 8px;
            }
            QToolButton:hover {
                background: #1AAB29;
            }
        """)


    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.window().isMaximized():
                self.window().showNormal()
                new_pos = event.globalPos() - QPoint(self.width()//2, self.height()//2)
                self.window().move(new_pos)
                self._dragging = True
                self._drag_position = event.globalPos() - self.window().frameGeometry().topLeft()
            else:
                self._dragging = True
                self._drag_position = event.globalPos() - self.window().frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.window().move(event.globalPos() - self._drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._dragging = False
        event.accept()





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

        self.is_dark_mode = True

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.main_widget = QWidget(self)
        self.main_widget.setObjectName("main_widget")
        self.setCentralWidget(self.main_widget)

        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self)
        self.title_bar.setObjectName("title_bar")
        self.main_layout.addWidget(self.title_bar)

        self.content_widget = QWidget()
        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.main_layout.addWidget(self.content_widget)

        self.side_bar_container = QWidget()
        self.side_bar_container_layout = QVBoxLayout(self.side_bar_container)
        self.side_bar_container.setObjectName("side_bar")
        self.side_bar_container_layout.setContentsMargins(0,10,0,0)

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
        self.sift_descriptors_window = SiftDescriptorsWindow(self)
        self.image_matching_window = ImageMatchingWindow(self)
        self.corner_detection_window = CornerDetectionWindow(self)
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
        self.content_layout.addWidget(self.side_bar_container)
        self.content_layout.addWidget(self.stackedWidget)


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

        self._margin = 8
        self._resizing = False
        self._resize_dir = None
        self.initial_pos = None

        self.apply_theme()




    def change_icon_color(self,icon_path,original_color,new_color):
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
                    icon = self.change_icon_color(icon_path,original_color="black",new_color= "#4379EE")
                else:
                    icon = self.change_icon_color(icon_path,original_color="black",new_color= "#2D60FF")
            else:
                icon = self.change_icon_color(icon_path,original_color="black", new_color="#B1B1B1")
            
            item.setIcon(icon)

        self.stackedWidget.setCurrentIndex(index)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode

        self.apply_theme()

        self.mode_toggle_signal.emit(self.is_dark_mode)
        self.on_sidebar_item_select(self.list_widget.currentRow())

    def apply_theme(self):
        if self.is_dark_mode:
            self.setStyleSheet(self.dark_mode_stylesheet())
            self.mode_toggle_button.setIcon(QIcon("assets/icons/moon.png"))
        else:
            self.setStyleSheet(self.light_mode_stylesheet())
            self.mode_toggle_button.setIcon(QIcon("assets/icons/sun.png"))

    def light_mode_stylesheet(self):
        return """
            #main_widget {
                background-color: #f5f7fa;
            }      
            #title_bar{
                background-color: white;
            }
            #title_bar_label { 
                font-size: 10pt; 
                color: #343C6A;
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
                color: #4379EE; 
                font-weight: bold;
                border-left: 4px solid #4379EE;
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
                background-color: #4379EE;
                color: #FFFFFF;
            }

            QPushButton#apply_button:hover {
                background-color: #FFFFFF;
                border: 1px solid #4379EE;
                color:#4379EE;
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
                selection-background-color: #4379EE;
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
            background-color: #4379EE; 
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
            background: #4379EE;
            min-height: 20px;
            border-radius: 4px;
        }

        QScrollBar::handle:vertical:hover {
            background: #4379EE;
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
            #title_bar{
                background-color: #273142;
            }
            #title_bar_label { 
                font-size: 10pt; 
                color: white;
            }
            QColorDialog {
                background-color: #273142;
                color: white;
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
                color: #4379EE; 
                font-weight: bold;
                border-left: 4px solid #4379EE;
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
                background-color: #4379EE;
            }

            QPushButton#mode_toggle_button:hover{
                padding: 8px 8px;
                border: 2px solid #4379EE;
                border-radius: 8px;
                background-color: #273142;
            }
                           
            QPushButton#apply_button {
                font-size: 18px;
                font-weight: bold;
                padding: 8px 25px;
                border-radius: 8px;
                background-color: #4379EE;
                color: #FFFFFF;
            }

            QPushButton#apply_button:hover {
                background-color: #1B2431;
                border: 1px solid #4379EE;
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
                border: 1px solid #4B5668;
                padding: 3px;
                font-size: 12px;
                color: #FFFFFF;
                background-color: #1B2431;
                selection-background-color: #4B5668;
            }
                           

            #spin_box::up-button {
                subcontrol-position: top right;
                width: 16px; /* Adjust width */
                background: #1B2431;
            }

            #spin_box::down-button {
                subcontrol-position: bottom right;
                width: 16px;
                background: #1B2431;

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
                border: 1px solid #4B5668;
                padding: 5px;
                color: #FFFFFF;
                background-color: #1B2431;
                selection-background-color: #FFFFFF;
            }
            QComboBox#combo_box QAbstractItemView {
                background-color: #1B2431;
                color: #FFFFFF;
                selection-background-color: #4B5668;
                selection-color: #FFFFFF;
            }
                           
            QComboBox#combo_box::drop-down {
                background-color: #1B2431; 
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
                border: 1px solid #4B5668;
                padding: 3px 10px;
                margin:0px;
                color: #FFFFFF;
                background-color: #1B2431;
                border-radius:4px;      
            }
                           
            #choose_color_button:hover {
                background-color: #4379EE; 
                color: #FFFFFF;
                border-color: #4379EE; 
            }              

            QScrollBar:vertical {
                background: #273142;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical {
                background: #4379EE;
                min-height: 20px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical:hover {
                background: #4379EE;
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


    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self._window_state_changed(self.windowState())
        super().changeEvent(event)
        event.accept()

    def _window_state_changed(self, state):
        # This method will now handle window state changes
        if state == Qt.WindowState.WindowMaximized:
            self.title_bar.normal_button.setVisible(True)
            self.title_bar.max_button.setVisible(False)
        else:
            self.title_bar.normal_button.setVisible(False)
            self.title_bar.max_button.setVisible(True)

    def _get_resize_direction(self, pos):
        rect = self.rect()
        x, y, w, h = pos.x(), pos.y(), rect.width(), rect.height()
        margin = self._margin

        directions = {
            "left": x <= margin,
            "right": x >= w - margin,
            "top": y <= margin,
            "bottom": y >= h - margin,
        }

        if directions["left"] and directions["top"]:
            return "top_left"
        if directions["right"] and directions["top"]:
            return "top_right"
        if directions["left"] and directions["bottom"]:
            return "bottom_left"
        if directions["right"] and directions["bottom"]:
            return "bottom_right"
        if directions["left"]:
            return "left"
        if directions["right"]:
            return "right"
        if directions["top"]:
            return "top"
        if directions["bottom"]:
            return "bottom"
        return None


    def _get_cursor_shape(self, direction):
        return {
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "top_left": Qt.CursorShape.SizeFDiagCursor,
            "top_right": Qt.CursorShape.SizeBDiagCursor,
            "bottom_left": Qt.CursorShape.SizeBDiagCursor,
            "bottom_right": Qt.CursorShape.SizeFDiagCursor,
        }.get(direction, Qt.CursorShape.ArrowCursor)


    def _resize_window(self, pos):
        if self.initial_pos is None:
            return
        diff = pos - self.initial_pos  
        geo = self.geometry()  

        if self._resize_dir == "left":
            geo.setLeft(geo.left() + diff.x())
        elif self._resize_dir == "right":
            geo.setRight(geo.right() + diff.x())
        elif self._resize_dir == "top":
            geo.setTop(geo.top() + diff.y())
        elif self._resize_dir == "bottom":
            geo.setBottom(geo.bottom() + diff.y())
        elif self._resize_dir == "top_left":
            geo.setTopLeft(geo.topLeft() + diff)
        elif self._resize_dir == "top_right":
            geo.setTopRight(geo.topRight() + diff)
        elif self._resize_dir == "bottom_left":
            geo.setBottomLeft(geo.bottomLeft() + diff)
        elif self._resize_dir == "bottom_right":
            geo.setBottomRight(geo.bottomRight() + diff)

        self.setGeometry(geo)
        self.initial_pos = pos  

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            direction = self._get_resize_direction(event.pos())
            if direction:
                self._resizing = True
                self._resize_dir = direction
                self.initial_pos = event.pos()  
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._resizing and self._resize_dir:
            self._resize_window(event.pos())  

        # direction = self._get_resize_direction(event.pos())
        # if direction:
        #     self.setCursor(self._get_cursor_shape(direction)) 
        # else:
        #     self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)
        event.accept()


    def mouseReleaseEvent(self, event):
        self._resizing = False
        self._resize_dir = None
        self.initial_pos = None
        super().mouseReleaseEvent(event)
        event.accept()
