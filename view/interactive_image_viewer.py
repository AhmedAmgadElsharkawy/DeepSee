from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from view.image_viewer import ImageViewer

class InteractiveImageViewer(ImageViewer):
    def __init__(self):
        super().__init__()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                self.show_image(file_path)

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.show_image(file_path)

    def show_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.setPixmap(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setStyleSheet("border: none;")
        self.setText("") 