from PyQt5.QtWidgets import QFileDialog

from view.image_viewer import ImageViewer

class InteractiveImageViewer(ImageViewer):
    def __init__(self):
        super().__init__(type = "input")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                self.load_image(file_path)

    def mouseDoubleClickEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.image_model.load_image(file_path=file_path)
        self.display_the_image_model()