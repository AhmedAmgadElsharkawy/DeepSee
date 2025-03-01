from PyQt5.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout
import pyqtgraph as pg
from PyQt5.QtGui import QFont

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.interactive_image_viewer import InteractiveImageViewer
from view.widget.image_viewer import ImageViewer

from controller.transformations_controller import TransformationsController

class TransformationsWindow(BasicStackedWindow):
    def __init__(self, main_window):
        super().__init__(main_window, "Transformations")

        self.image_viewers_container.deleteLater()

        self.transformation_type_custom_combo_box = CustomComboBox(label= "Transformation Type",combo_box_items_list=["Grayscale","Equalization","Normalization"])
        self.inputs_container_layout.addWidget(self.transformation_type_custom_combo_box)

        self.input_image_viewer = InteractiveImageViewer()
        self.output_image_viewer = ImageViewer()

        self.original_image_row = QWidget()
        self.original_image_row.setObjectName("original_image_row")
        self.original_image_row_layout = QHBoxLayout(self.original_image_row)
        self.original_image_row_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.addWidget(self.original_image_row)
        self.original_image_row_layout.addWidget(self.input_image_viewer,1)

        self.transformed_image_row = QWidget()
        self.transformed_image_row.setObjectName("transformed_image_row")
        self.transformed_image_row_layout = QHBoxLayout(self.transformed_image_row)
        self.transformed_image_row_layout.setContentsMargins(0,0,0,0)
        self.main_widget_layout.addWidget(self.transformed_image_row)
        self.transformed_image_row_layout.addWidget(self.output_image_viewer,1)
        
        

        
        self.orignal_image_graphs_container = QWidget()
        self.orignal_image_graphs_container.setObjectName("orignal_image_graphs_container")
        self.orignal_image_graphs_container_layout = QVBoxLayout(self.orignal_image_graphs_container)
        self.orignal_image_graphs_container_layout.setContentsMargins(0,0,0,0)
        self.original_image_row_layout.addWidget(self.orignal_image_graphs_container,2)

        self.transformed_image_graphs_container = QWidget()
        self.transformed_image_graphs_container.setObjectName("transformed_image_graphs_container")
        self.transformed_image_graphs_container_layout = QVBoxLayout(self.transformed_image_graphs_container)
        self.transformed_image_graphs_container_layout.setContentsMargins(0,0,0,0)
        self.transformed_image_row_layout.addWidget(self.transformed_image_graphs_container,2)

        self.orignal_image_histogram_graph = pg.PlotWidget(title="Original Histogram")
        self.orignal_image_pdf_graph = pg.PlotWidget(title="Original PDF")
        self.orignal_image_cdf_graph = pg.PlotWidget(title="Original CDF")
        self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_histogram_graph)
        # self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_pdf_graph)
        self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_cdf_graph)

        self.transformed_image_histogram_graph = pg.PlotWidget(title="Transformed Histogram")
        self.transformed_image_pdf_graph = pg.PlotWidget(title="Transformed PDF")
        self.transformed_image_cdf_graph = pg.PlotWidget(title="Transformed CDF")
        self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_histogram_graph)
        # self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_pdf_graph)
        self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_cdf_graph)

        for graph in [self.orignal_image_histogram_graph, self.orignal_image_pdf_graph, self.orignal_image_cdf_graph,self.transformed_image_cdf_graph,self.transformed_image_histogram_graph,self.transformed_image_pdf_graph]:
            graph.showGrid(x=True, y=True)
            graph.setBackground("#FFFFFF")
            graph.getAxis("left").setPen("#DFE5EE")
            graph.getAxis("bottom").setPen("#DFE5EE")
            graph.getPlotItem().titleLabel.item.setFont(QFont("Arial"))

        self.transformations_controller = TransformationsController(self)



        


        

    


