from PyQt5.QtWidgets import QLabel, QPushButton, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QSize

import pyqtgraph as pg
import numpy as np
import cv2

from model.image_model import ImageModel

class ImageViewer(pg.ImageView):
    def __init__(self, type="output"):
        super().__init__()
        self.image_model = ImageModel()
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        from view.main_window import MainWindow
        self.main_window = MainWindow()


        self.getView().setMenuEnabled(False)
        self.getView().setBackgroundColor(QColor("white"))

        self.save_image_button = None

        self.temp_label = QLabel(parent=self)
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setFont(QFont("Inter", 10, QFont.Bold))
        self.temp_label.setObjectName("temp_label")

        self.temp_label.setGeometry(0, 0, self.width(), self.height())

        if type == "input":
            self.temp_label_placeholder_text = "Double click, or drop image here\n\nAllowed Files: PNG, JPG, JPEG BMP files"
        else:
            self.temp_label_placeholder_text = "Processed image will appear here"
            # self.save_image_button = QPushButton(self)
            # self.save_image_button.setIcon(QIcon("assets/icons/save_icon.png"))
            # self.save_image_button.setIconSize(QSize(30, 30))
            # self.save_image_button.setGeometry(20, 20, 30, 30)
            # self.save_image_button.setCursor(Qt.CursorShape.PointingHandCursor)
            # self.save_image_button.clicked.connect(self.on_save_image_click)
            # self.save_image_button.setEnabled(False)
            # self.save_image_button.setObjectName("save_image_button")

        self.temp_label.setText(self.temp_label_placeholder_text)

    def contextMenuEvent(self, event):
        if self.image_model.image_matrix is None:
            return 

        menu = QMenu(self)

        move_menu = menu.addMenu("Move To")
        
        save_image_action = QAction("Save")
        save_image_action.triggered.connect(self.on_save_image_click)
        menu.addAction(save_image_action)

        noise_action = QAction("Noise", self)
        filters_action = QAction("Filters", self)
        threshold_action = QAction("Thresholding", self)
        edge_action = QAction("Edge Detection", self)
        transformations_action = QAction("Transformations", self)

        first_viewer_action = QAction("First Image Viewer", self)
        second_viewer_action = QAction("Second Image Viewer", self)

        move_menu.addAction(noise_action)
        move_menu.addAction(filters_action)
        move_menu.addAction(threshold_action)
        move_menu.addAction(edge_action)
        move_menu.addAction(transformations_action)

        hybrid_menu = move_menu.addMenu("Hybrid Image")
        hybrid_menu.addAction(first_viewer_action)
        hybrid_menu.addAction(second_viewer_action)

        noise_action.triggered.connect(self.move_to_noise)
        filters_action.triggered.connect(self.move_to_filters)
        threshold_action.triggered.connect(self.move_to_thresholding)
        edge_action.triggered.connect(self.move_to_edge_detection)
        transformations_action.triggered.connect(self.move_to_transformations)
        first_viewer_action.triggered.connect(self.move_to_first_viewer)
        second_viewer_action.triggered.connect(self.move_to_second_viewer)

        menu.exec_(event.globalPos())

    def move_to_noise(self):
        self.main_window.nosie_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.nosie_window.output_image_viewer.reset()

    def move_to_filters(self):
        self.main_window.filters_window.output_image_viewer.reset()
        print(self.image_model.image_matrix.shape)
        self.main_window.filters_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        print(self.main_window.filters_window.input_image_viewer.image_model.get_gray_image_matrix().shape)
        print(self.main_window.filters_window.input_image_viewer.image_model.get_image_matrix().shape)




    def move_to_thresholding(self):
        self.main_window.thresholding_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.thresholding_window.output_image_viewer.reset()

    def move_to_edge_detection(self):
        self.main_window.edge_detection_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.edge_detection_window.output_image_viewer.reset()

    def move_to_transformations(self):
        self.main_window.transformations_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.transformations_window.output_image_viewer.reset()
        self.main_window.transformations_window.orignal_image_histogram_graph.clear()
        self.main_window.transformations_window.orignal_image_cdf_graph.clear()
        self.main_window.transformations_window.orignal_image_pdf_graph.clear()
        self.main_window.transformations_window.transformed_image_histogram_graph.clear()
        self.main_window.transformations_window.transformed_image_cdf_graph.clear()
        self.main_window.transformations_window.transformed_image_pdf_graph.clear()

    def move_to_first_viewer(self):
        self.main_window.hybrid_image_widnow.first_original_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.hybrid_image_widnow.first_filtered_image_viewer.reset()
        self.main_window.hybrid_image_widnow.hybrid_image_viewer.reset()

    def move_to_second_viewer(self):
        self.main_window.hybrid_image_widnow.second_original_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.hybrid_image_widnow.second_filtered_image_viewer.reset()
        self.main_window.hybrid_image_widnow.hybrid_image_viewer.reset()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.temp_label.setGeometry(0, 0, self.width(), self.height())
        # if self.save_image_button is not None:
        #     self.save_image_button.setGeometry(20, 20, 30, 30)

    def display_image_matrix(self, image_matrix):
        self.temp_label.hide()
        if self.save_image_button is not None:
            self.save_image_button.setEnabled(True)
        matrix_to_display = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        matrix_to_display = np.transpose(matrix_to_display, (1, 0, 2))
        self.setImage(matrix_to_display)

    def display_image_matrix2(self, image_matrix):
        self.temp_label.hide()
        if self.save_image_button != None:
            self.save_image_button.setEnabled(True)
        matrix_to_display = np.transpose(image_matrix)
        self.setImage(matrix_to_display)

    def display_and_set_image_matrix(self, image_matrix):
        self.display_image_matrix(image_matrix)
        self.image_model.set_image_matrix(image_matrix)

    def display_the_image_model(self):
        self.display_image_matrix(self.image_model.get_image_matrix())

    def on_save_image_click(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp);;All Files (*)")
        if save_path:
            self.image_model.save_image(save_path=save_path)

    def reset(self):
        self.image_model.reset() 
        self.temp_label.setText(self.temp_label_placeholder_text)  
        self.temp_label.show()  
        self.clear()
        if self.save_image_button is not None:
            self.save_image_button.setEnabled(False)

