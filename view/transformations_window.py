from PyQt5.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout

from view.basic_stacked_window import BasicStackedWindow
from view.custom_combo_box import CustomComboBox
import pyqtgraph as pg

class TransformationsWindow(BasicStackedWindow):
    def __init__(self):
        super().__init__("Transformations")

        self.transformation_type_custom_combo_box = CustomComboBox(label= "Transformation Type",combo_box_items_list=["Grayscale","Equalization","Normalization"])
        self.inputs_container_layout.addWidget(self.transformation_type_custom_combo_box)

        self.graphs_widget_container = QWidget()
        self.graphs_widget_container_layout = QHBoxLayout(self.graphs_widget_container)
        self.main_widget_layout.addWidget(self.graphs_widget_container)
        self.graphs_widget_container_layout.setContentsMargins(0,0,0,0)
        
        self.orignal_image_graphs_container = QWidget()
        self.orignal_image_graphs_container_layout = QVBoxLayout(self.orignal_image_graphs_container)
        self.orignal_image_graphs_container_layout.setContentsMargins(0,0,0,0)
        self.graphs_widget_container_layout.addWidget(self.orignal_image_graphs_container)

        self.transformed_image_graphs_container = QWidget()
        self.transformed_image_graphs_container_layout = QVBoxLayout(self.transformed_image_graphs_container)
        self.transformed_image_graphs_container_layout.setContentsMargins(0,0,0,0)
        self.graphs_widget_container_layout.addWidget(self.transformed_image_graphs_container)

        self.orignal_image_histogram_graph = pg.PlotWidget(title="Original Histogram")
        self.orignal_image_pdf_graph = pg.PlotWidget(title="Original PDF")
        self.orignal_image_cdf_graph = pg.PlotWidget(title="Original CDF")
        self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_histogram_graph)
        self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_pdf_graph)
        self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_cdf_graph)

        self.transformed_image_histogram_graph = pg.PlotWidget(title="Transformed Histogram")
        self.transformed_image_pdf_graph = pg.PlotWidget(title="Transformed PDF")
        self.transformed_image_cdf_graph = pg.PlotWidget(title="Transformed CDF")
        self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_histogram_graph)
        self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_pdf_graph)
        self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_cdf_graph)



        


        

    


