import cv2
import numpy as np
class FaceDetectionController():
    def __init__(self,Face_detection_and_recognition_window = None):
        self.Face_detection_and_recognition_window = Face_detection_and_recognition_window
        if self.Face_detection_and_recognition_window:
            self.Face_detection_and_recognition_window.apply_button.clicked.connect(self.apply_face_detection)


    def apply_face_detection(self):
        if self.Face_detection_and_recognition_window.face_analysis_type_custom_combo_box.current_text() == "Face Detection":
            image = self.Face_detection_and_recognition_window.input_image_viewer.image_model.get_image_matrix()
            output_image = self.face_detection(image)
            self.Face_detection_and_recognition_window.output_image_viewer.display_and_set_image_matrix(output_image)
            print("Face Detection")
        else:
            pass

    def face_detection(self,image):
        # Load the pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return image

        