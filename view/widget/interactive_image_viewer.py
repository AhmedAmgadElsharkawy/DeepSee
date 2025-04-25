from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsLineItem
from PyQt5.QtGui import QPen

from view.widget.image_viewer import ImageViewer


class InteractiveImageViewer(ImageViewer):
    def __init__(self, custom_placeholder=None):
        super().__init__(type="input", custom_placeholder=custom_placeholder)
        self.setAcceptDrops(True)
        self.markers_positions = []
        self.marker_items = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                self.load_image(file_path)

    def mouseDoubleClickEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.reset()
        self.image_model.load_image(file_path=file_path)
        self.display_image_matrix(self.image_model.get_image_matrix())

    def handle_mouse_click(self, event):
        if event.button() == Qt.LeftButton and self.image_model.image_matrix is not None:
            mouse_point = self.getView().mapSceneToView(event.scenePos())
            x, y = mouse_point.x(), mouse_point.y()

            image_matrix = self.image_model.image_matrix
            height, width = image_matrix.shape[:2]

            if 0 <= x < width and 0 <= y < height:
                self.add_x_marker(x, y)

    def add_x_marker(self, x, y, size=6):
        line1 = QGraphicsLineItem(x - size/2, y - size/2, x + size/2, y + size/2)
        line2 = QGraphicsLineItem(x - size/2, y + size/2, x + size/2, y - size/2)

        pen = QPen(Qt.red)
        pen.setWidthF(1.5)
        line1.setPen(pen)
        line2.setPen(pen)

        self.getView().addItem(line1)
        self.getView().addItem(line2)

        self.marker_items.extend([line1, line2])
        self.markers_positions.append({"x" : int(x), 'y' : int(y)})

    def reset_markers(self):
        for item in self.marker_items:
            self.getView().removeItem(item)
        self.marker_items.clear()
        self.markers_positions.clear()


    def reset(self):
        super().reset()  
        self.reset_markers()


    def enable_add_marker(self, enabled: bool):
        if enabled:
            self.getView().scene().sigMouseClicked.connect(self.handle_mouse_click)
        elif not enabled:
            self.getView().scene().sigMouseClicked.disconnect(self.handle_mouse_click)
            self.reset_markers()
