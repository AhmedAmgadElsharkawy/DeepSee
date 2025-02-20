import pyqtgraph as pg
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt

class ImageViewer(pg.ImageView):
    def __init__(self,type = "output"):
        super().__init__()

        self.ui.histogram.hide()  
        self.ui.roiBtn.hide()     
        self.ui.menuBtn.hide()    

        self.getView().setBackgroundColor(QColor(45, 45, 45)) 

        if type == "input":
            self.temp_label_placeholder_text = "Double click, or drop your image here\n\nAllowed Files: PNG, JPG, JPEG files"
        else:
            self.temp_label_placeholder_text = "Processed image will appear here"
        self.temp_label = QLabel(self.temp_label_placeholder_text, self)
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.temp_label.setStyleSheet("""
            color: #A0A0A0;
            background: transparent;
        """)

        self.temp_label.setGeometry(0, 0, self.width(), self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.temp_label.setGeometry(0, 0, self.width(), self.height())

    def setImage(self, img, **kwargs):
        self.temp_label.hide()  
        super().setImage(img, **kwargs)
