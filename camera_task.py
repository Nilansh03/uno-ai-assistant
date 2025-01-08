import cv2
import os
from ultralytics import YOLO
import mediapipe as mp
import threading
from queue import Queue


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Prepare the camera
        print("Camera warming up ...")

        self.KNOWN_FACES_DIR = "known_faces"
        self.NEW_FACES_DIR = "new_faces"
        os.makedirs(self.KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(self.NEW_FACES_DIR, exist_ok=True)

        self.object_detection_model = YOLO("yolov8s.pt", verbose=False)

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        self.known_faces_encodings = []
        self.known_faces_names = []
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.command_queue = Queue()  # Queue to hold commands

        # Load known faces
        self.load_known_faces()

        # Start the frame capture thread
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()

    def load_known_faces(self):
        # Load known faces from the known_faces directory
        for filename in os.listdir(self.KNOWN_FACES_DIR):
            filepath = os.path.join(self.KNOWN_FACES_DIR, filename)
            image = cv2.imread(filepath, 0)
            self.known_faces_encodings.append(image)
            self.known_faces_names.append(os.path.splitext(filename)[0])

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def recognize_person(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                face_roi = frame[y:y + h, x:x + w]
                name = "Unknown"
                for i, known_face in enumerate(self.known_faces_encodings):
                    res = cv2.matchTemplate(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY),
                                            known_face, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > 0.6:
                        name = self.known_faces_names[i]
                        break

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def detect_objects(self, frame):
        results = self.object_detection_model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            label = self.object_detection_model.names[int(result.cls[0])]
            confidence = result.conf[0].item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def release_camera(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# if __name__ == '__main__':
#     main()
