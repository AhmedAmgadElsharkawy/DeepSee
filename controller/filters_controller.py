class FiltersController():
    def __init__(self,filters_window):
        self.filters_window = filters_window
        self.filters_window.apply_button.clicked.connect(self.apply_filter)

    def apply_filter(self):
        print("applied")