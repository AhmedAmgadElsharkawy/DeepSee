from PyQt5.QtWidgets import QLabel,QPushButton,QFileDialog
from PyQt5.QtGui import QColor, QFont,QIcon
from PyQt5.QtCore import Qt,QSize
import pyqtgraph as pg
import numpy as np
import cv2

from model.image_model import ImageModel

class ImageViewer(pg.ImageView):
    def __init__(self,type = "output"):
        super().__init__()
        self.image_model = ImageModel()
        self.ui.histogram.hide()  
        self.ui.roiBtn.hide()     
        self.ui.menuBtn.hide()    

        self.getView().setBackgroundColor(QColor("white")) 

        self.save_image_button = None

        self.temp_label = QLabel(parent= self)
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.temp_label.setObjectName("temp_label")

        self.temp_label.setGeometry(0, 0, self.width(), self.height())

        if type == "input":
            self.temp_label_placeholder_text = "Double click, or drop image here\n\nAllowed Files: PNG, JPG, JPEG files"
        else:
            self.temp_label_placeholder_text = "Processed image will appear here"
            self.save_image_button = QPushButton(self)
            self.save_image_button.setIcon(QIcon("assets/icons/save_icon.png") )
            self.save_image_button.setIconSize(QSize(30, 30))
            self.save_image_button.setGeometry(20,20,30,30)
            self.save_image_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.save_image_button.clicked.connect(self.on_save_image_click)
            self.save_image_button.setEnabled(False)
            self.save_image_button.setObjectName("save_image_button")

        self.temp_label.setText(self.temp_label_placeholder_text)

        self.setStyleSheet("""
                #temp_label{
                    color: #A0A0A0;
                    background: transparent;     
                }
                #save_image_button{
                    border: none;
                    background-color: transparent;   
                }
                #save_image_button:hover{
                    background-color: none;
                    }
                """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.temp_label.setGeometry(0, 0, self.width(), self.height())
        if self.save_image_button != None:
            self.save_image_button.setGeometry(20,20,30,30)


    def display_image_matrix(self, image_matrix):
        self.temp_label.hide() 
        if self.save_image_button != None:
            self.save_image_button.setEnabled(True)
        matrix_to_display = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        matrix_to_display = np.transpose(matrix_to_display, (1, 0, 2))
        self.setImage(matrix_to_display)
    def display_image_matrix2(self, image_matrix):
        self.temp_label.hide()
        if self.save_image_button != None:
            self.save_image_button.setVisible(True)
        matrix_to_display = np.transpose(image_matrix)
        self.setImage(matrix_to_display)

    def display_and_set_image_matrix(self,image_matrix):
        self.display_image_matrix(image_matrix)
        self.image_model.set_image_matrix(image_matrix)
        
    def display_the_image_model(self):
        self.display_image_matrix(self.image_model.get_image_matrix())

    def on_save_image_click(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)")
        if save_path:
            self.image_model.save_image(save_path=save_path)