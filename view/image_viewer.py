from PyQt5.QtWidgets import  QLabel
from PyQt5.QtGui import  QFont
from PyQt5.QtCore import Qt

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()

        self.setText("Click, or drop your images here\n\nAllowed Files: PNG, JPG, JPEG files")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2E2E2E;
                color: #A0A0A0;
                font-size: 16px;
                border: 2px dashed #555;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        self.setFont(QFont("Arial", 12, QFont.Bold))
        self.setAcceptDrops(True)