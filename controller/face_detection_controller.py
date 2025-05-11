import cv2
import numpy as np
import multiprocessing as mp
from PyQt5.QtCore import QThread, pyqtSignal



def face_detection_process(image, queue = None):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if queue is None:
        return image
    else:
        queue.put(image)



class FaceDetectionWorker(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        queue = mp.Queue()
        
        process = mp.Process(target=face_detection_process, args=(self.params['image'], queue))
        process.start()

        while True:
            if not queue.empty():
                result = queue.get()
                self.result_ready.emit(result)
                break
            self.msleep(50)

        process.join()

class FaceDetectionController():
    def __init__(self,Face_detection_and_recognition_window = None):
        self.Face_detection_and_recognition_window = Face_detection_and_recognition_window
        if self.Face_detection_and_recognition_window:
            self.Face_detection_and_recognition_window.apply_button.clicked.connect(self.apply_face_detection)


    def apply_face_detection(self):
        if self.Face_detection_and_recognition_window.face_analysis_type_custom_combo_box.current_text() == "Face Detection":
            image = self.Face_detection_and_recognition_window.input_image_viewer.image_model.get_image_matrix()

            self.Face_detection_and_recognition_window.output_image_viewer.show_loading_effect()
            self.Face_detection_and_recognition_window.controls_container.setEnabled(False)
            self.Face_detection_and_recognition_window.image_viewers_container.setEnabled(False)

            params = {"image" : image}
            self.worker = FaceDetectionWorker(params)
            self.worker.result_ready.connect(self._on_result)
            self.worker.start()

        else:
            pass

    
    def _on_result(self,result_image):
        self.Face_detection_and_recognition_window.output_image_viewer.hide_loading_effect()
        self.Face_detection_and_recognition_window.controls_container.setEnabled(True)
        self.Face_detection_and_recognition_window.image_viewers_container.setEnabled(True)

        self.Face_detection_and_recognition_window.output_image_viewer.display_and_set_image_matrix(result_image)
        self.Face_detection_and_recognition_window.show_toast(title = "Success!", text = "Face detection is complete.")

        