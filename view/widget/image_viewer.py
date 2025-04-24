from PyQt5.QtWidgets import QLabel, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QColor, QFont,QMovie
from PyQt5.QtCore import Qt

import pyqtgraph as pg
import numpy as np
import cv2

from model.image_model import ImageModel

class ImageViewer(pg.ImageView):
    def __init__(self, type="output",custom_placeholder = None):
        super().__init__()
        self.image_model = ImageModel()
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        from view.main_window import MainWindow
        self.main_window = MainWindow()


        self.getView().setMenuEnabled(False)

        self.save_image_button = None

        self.temp_label = QLabel(parent=self)
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setFont(QFont("Inter", 10, QFont.Bold))
        self.temp_label.setObjectName("temp_label")

        self.temp_label.setGeometry(0, 0, self.width(), self.height())
        
        if custom_placeholder:
            self.temp_label_placeholder_text = custom_placeholder
        elif type == "input":
            self.temp_label_placeholder_text = "Double click, or drop image here\n\nAllowed Files: PNG, JPG, JPEG BMP files"
        else:
            self.temp_label_placeholder_text = "Processed image will appear here"

        self.temp_label.setText(self.temp_label_placeholder_text)


        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setGeometry(0, 0, self.width(), self.height())
        self.loading_label.setObjectName("loading_label")

        self.movie = QMovie("assets/icons/loading.gif")
        self.loading_label.setMovie(self.movie)

        self.choose_mode(self.main_window.is_dark_mode)
        self.main_window.mode_toggle_signal.connect(self.choose_mode)


    def show_loading_effect(self):
        self.clear()
        self.temp_label.hide()
        self.loading_label.show()
        self.movie.start()

    def hide_loading_effect(self):
        self.loading_label.hide()

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
        hough_action = QAction("Hough Transform", self)
        active_contours_action = QAction("Active Contours", self)
        corner_detection_action = QAction("Corner Detection", self)
        sift_descriptors_action = QAction("Sift Descriptors")
        segmentation_action = QAction("Segmentation",self)
        face_detection_and_recognition_action = QAction("Face Detection & Recognition")

        hybrid_image_first_viewer_action = QAction("First Image", self)
        hybrid_image_second_viewer_action = QAction("Second Image", self)

        image_matching_input_image_viewer_action = QAction("First Image", self)
        image_matching_input_image2_viewer_action = QAction("Second Image", self)

        sift_descriptors_detect_keypoints_action = QAction("Detect Keypoints", self)

        move_menu.addAction(noise_action)
        move_menu.addAction(filters_action)
        move_menu.addAction(threshold_action)
        move_menu.addAction(edge_action)
        move_menu.addAction(transformations_action)
        move_menu.addAction(hough_action)
        move_menu.addAction(active_contours_action)
        move_menu.addAction(corner_detection_action)
        move_menu.addAction(sift_descriptors_action)
        move_menu.addAction(segmentation_action)
        move_menu.addAction(face_detection_and_recognition_action)

        hybrid_menu = move_menu.addMenu("Hybrid Image")
        hybrid_menu.addAction(hybrid_image_first_viewer_action)
        hybrid_menu.addAction(hybrid_image_second_viewer_action)

        image_matching_menu = move_menu.addMenu("Image Matching")
        image_matching_menu.addAction(image_matching_input_image_viewer_action)
        image_matching_menu.addAction(image_matching_input_image2_viewer_action)


        noise_action.triggered.connect(self.move_to_noise)
        filters_action.triggered.connect(self.move_to_filters)
        threshold_action.triggered.connect(self.move_to_thresholding)
        edge_action.triggered.connect(self.move_to_edge_detection)
        transformations_action.triggered.connect(self.move_to_transformations)
        hybrid_image_first_viewer_action.triggered.connect(self.move_to_first_viewer)
        hybrid_image_second_viewer_action.triggered.connect(self.move_to_second_viewer)
        hough_action.triggered.connect(self.move_to_hough_transform_viewer)
        active_contours_action.triggered.connect(self.move_to_active_contours_viewer)
        corner_detection_action.triggered.connect(self.move_to_corner_detection_viewer)
        image_matching_input_image_viewer_action.triggered.connect(self.move_to_image_matching_window_image_viewer)
        image_matching_input_image2_viewer_action.triggered.connect(self.move_to_image_matching_window_image2_viewer)
        sift_descriptors_detect_keypoints_action.triggered.connect(self.move_to_sift_descriptors_window)
        segmentation_action.triggered.connect(self.move_to_segmentation)
        face_detection_and_recognition_action.triggered.connect(self.move_to_face_detection_and_recognition)
        sift_descriptors_action.triggered.connect(self.move_to_sift_descriptors_window)

        menu.exec_(event.globalPos())

    def move_to_noise(self):
        self.main_window.nosie_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.nosie_window.output_image_viewer.reset()

    def move_to_filters(self):
        self.main_window.filters_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.filters_window.output_image_viewer.reset()

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
        self.main_window.hybrid_image_window.first_original_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.hybrid_image_window.first_filtered_image_viewer.reset()
        self.main_window.hybrid_image_window.hybrid_image_viewer.reset()

    def move_to_second_viewer(self):
        self.main_window.hybrid_image_window.second_original_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.hybrid_image_window.second_filtered_image_viewer.reset()
        self.main_window.hybrid_image_window.hybrid_image_viewer.reset()

    def move_to_hough_transform_viewer(self):
        self.main_window.hough_transform_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.hough_transform_window.output_image_viewer.reset()
    
    def move_to_active_contours_viewer(self):
        self.main_window.active_contours_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.active_contours_window.output_image_viewer.reset()
    
    def move_to_image_matching_window_image_viewer(self):
        self.main_window.image_matching_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.image_matching_window.output_image_viewer.reset()

    def move_to_image_matching_window_image2_viewer(self):
        self.main_window.image_matching_window.input_img2_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.image_matching_window.output_image_viewer.reset()
    
    def move_to_corner_detection_viewer(self):
        self.main_window.corner_detection_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.corner_detection_window.output_image_viewer.reset()
        
    def move_to_sift_descriptors_window(self):
        self.main_window.sift_descriptors_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.sift_descriptors_window.output_image_viewer.reset()
    
    def move_to_segmentation(self):
        self.main_window.segmentation_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.segmentation_window.output_image_viewer.reset()
    
    def move_to_face_detection_and_recognition(self):
        self.main_window.Face_detection_and_recognition_window.input_image_viewer.display_and_set_image_matrix(self.image_model.image_matrix)
        self.main_window.Face_detection_and_recognition_window.output_image_viewer.reset()
    


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.temp_label.setGeometry(0, 0, self.width(), self.height())
        self.loading_label.setGeometry(0, 0, self.width(), self.height())

    def display_image_matrix(self, image_matrix):
        self.temp_label.hide()
        self.loading_label.hide()
        matrix_to_display = None
        if not self.image_model.is_grayscale(image_matrix):
            matrix_to_display = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
            matrix_to_display = np.transpose(matrix_to_display, (1, 0, 2))
        else:
            matrix_to_display = np.transpose(image_matrix)
        self.setImage(matrix_to_display)

    def display_and_set_image_matrix(self, image_matrix):
        self.display_image_matrix(image_matrix)
        self.image_model.set_image_matrix(image_matrix)

    def on_save_image_click(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp);;All Files (*)")
        if save_path:
            self.image_model.save_image(save_path=save_path)

    def reset(self):
        self.image_model.reset() 
        self.temp_label.setText(self.temp_label_placeholder_text)  
        self.temp_label.show()  
        self.clear()


    def choose_mode(self,is_dark_mode):
        if is_dark_mode:
            self.dark_mode()
        else:
            self.light_mode()
        

    def dark_mode(self):
        self.getView().setBackgroundColor(QColor("#273142"))

    def light_mode(self):
        self.getView().setBackgroundColor(QColor("white"))

