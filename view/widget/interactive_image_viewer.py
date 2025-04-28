from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QGraphicsLineItem
from PyQt5.QtGui import QPen

from view.widget.image_viewer import ImageViewer


class InteractiveImageViewer(ImageViewer):
    def __init__(self, custom_placeholder=None):
        super().__init__(type="input", custom_placeholder=custom_placeholder)
        self.setAcceptDrops(True)
        self.markers_positions = []
        self.marker_items = []
        self.add_markers_connected = False
        self.just_removed_item = False

        self.just_double_clicked = False
        self.click_timer = None  # To hold the timer for detecting double-click

    def mouseDoubleClickEvent(self, event):
        # print("mouseDoubleClickEvent_start")
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.just_double_clicked = True
            self.load_image(file_path)
            # print("mouseDoubleClickEvent_end")


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                self.load_image(file_path)


    def load_image(self, file_path):
        # print("load start")
        self.reset()
        self.image_model.load_image(file_path=file_path)
        self.display_image_matrix(self.image_model.get_image_matrix())
        # print("load end")

    def handle_mouse_click(self, event):
        if self.just_double_clicked:
            self.just_double_clicked = False
            return

        if self.image_model.image_matrix is None:
            return

        mouse_point = self.getView().mapSceneToView(event.scenePos())
        x, y = mouse_point.x(), mouse_point.y()

        if event.button() == Qt.RightButton:
            for i, marker in enumerate(self.markers_positions):
                marker_x, marker_y = marker["x"], marker["y"]
                distance = ((x - marker_x) ** 2 + (y - marker_y) ** 2) ** 0.5
                if distance <= 5: 
                    self.remove_marker(i)
                    self.just_removed_item = True
                    return  
        elif event.button() == Qt.LeftButton:
            height, width = self.image_model.image_matrix.shape[:2]
            if 0 <= x < width and 0 <= y < height:
                self.add_x_marker(x, y)

    def remove_marker(self, index):
        line1 = self.marker_items.pop(index * 2)
        line2 = self.marker_items.pop(index * 2)
        self.getView().removeItem(line1)
        self.getView().removeItem(line2)
        self.markers_positions.pop(index)


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
        print(self.markers_positions)

    def reset_markers(self):
        for item in self.marker_items:
            self.getView().removeItem(item)
        self.marker_items.clear()
        self.markers_positions.clear()


    def reset(self):
        super().reset()  
        self.reset_markers()

    def enable_add_marker(self, enabled: bool):
        scene = self.getView().scene()
        if enabled and not self.add_markers_connected:
            scene.sigMouseClicked.connect(self.handle_mouse_click)
            self.add_markers_connected = True
        elif not enabled and self.add_markers_connected:
            scene.sigMouseClicked.disconnect(self.handle_mouse_click)
            self.add_markers_connected = False
            self.reset_markers()

    def get_markers_positions(self):
        return self.markers_positions
    

    def contextMenuEvent(self, event):
        if self.image_model.image_matrix is None or self.just_removed_item:
            self.just_removed_item = False
            return 
        self.display_context_menu(event)
