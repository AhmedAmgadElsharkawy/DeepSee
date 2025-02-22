from PyQt5.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout
import pyqtgraph as pg
from PyQt5.QtGui import QFont

from view.window.basic_stacked_window import BasicStackedWindow
from view.widget.custom_combo_box import CustomComboBox
from view.widget.interactive_image_viewer import InteractiveImageViewer
from view.widget.image_viewer import ImageViewer

from controller.transformations_controller import TransformationsController

class TransformationsWindow(BasicStackedWindow):
    def __init__(self):
        super().__init__("Transformations")

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
        # self.orignal_image_graphs_container_layout.setContentsMargins(0,0,0,0)
        self.original_image_row_layout.addWidget(self.orignal_image_graphs_container,2)

        self.transformed_image_graphs_container = QWidget()
        self.transformed_image_graphs_container.setObjectName("transformed_image_graphs_container")
        self.transformed_image_graphs_container_layout = QVBoxLayout(self.transformed_image_graphs_container)
        # self.transformed_image_graphs_container_layout.setContentsMargins(0,0,0,0)
        self.transformed_image_row_layout.addWidget(self.transformed_image_graphs_container,2)

        self.orignal_image_histogram_graph = pg.PlotWidget(title="Original Histogram")
        self.orignal_image_pdf_graph = pg.PlotWidget(title="Original PDF")
        self.orignal_image_cdf_graph = pg.PlotWidget(title="Original CDF")
        self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_histogram_graph)
        self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_pdf_graph)
        # self.orignal_image_graphs_container_layout.addWidget(self.orignal_image_cdf_graph)

        self.transformed_image_histogram_graph = pg.PlotWidget(title="Transformed Histogram")
        self.transformed_image_pdf_graph = pg.PlotWidget(title="Transformed PDF")
        self.transformed_image_cdf_graph = pg.PlotWidget(title="Transformed CDF")
        self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_histogram_graph)
        self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_pdf_graph)
        # self.transformed_image_graphs_container_layout.addWidget(self.transformed_image_cdf_graph)

        for graph in [self.orignal_image_histogram_graph, self.orignal_image_pdf_graph, self.orignal_image_cdf_graph,self.transformed_image_cdf_graph,self.transformed_image_histogram_graph,self.transformed_image_pdf_graph]:
            graph.showGrid(x=True, y=True)
            graph.setBackground("#F5F5F5")
            graph.getAxis("left").setPen("#333")
            graph.getAxis("bottom").setPen("#333")
            graph.getPlotItem().titleLabel.item.setFont(QFont("Arial"))

        self.transformations_controller = TransformationsController(self)


        self.setStyleSheet("""
            #transformed_image_graphs_container{   
                border:2px solid gray;
                border-radius:6px;
                           }
            #orignal_image_graphs_container{
                border:2px solid gray;
                border-radius:6px;           
                }
            #apply_button {
            font-size: 18px;
            font-weight: bold;
            padding: 8px 25px;
            border: 2px solid #888888;
            border-radius: 8px;
            background-color: #E0E0E0;
            color: #333333;
        }
        
        #apply_button:hover {
            background-color: #D0D0D0;
            border-color: #777777;
        }

        #apply_button:pressed {
            background-color: #B0B0B0;
            border-color: #666666;
        }

        #apply_button:disabled {
            background-color: #C0C0C0;
            border-color: #A0A0A0;
            color: #666666;
        }
                           
        #header_label{
            color: #333;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;           
            }
        """)


        


        

    


