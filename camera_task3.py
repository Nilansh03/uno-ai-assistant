import cv2
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from datetime import datetime

class LiveCameraFeed:
    def __init__(self):
        # Paths for storing faces
        self.KNOWN_FACES_DIR = "known_faces"
        self.NEW_FACES_DIR = "new_faces"
        os.makedirs(self.KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(self.NEW_FACES_DIR, exist_ok=True)

        # Load YOLOv5s for object detection
        self.object_detection_model = YOLO("yolov8n.pt")

        # Initialize Mediapipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not access the camera.")
            exit()

        # Initialize the variables that will be used across methods
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.running_tasks = []

        # Load known faces
        self.load_known_faces()

    def load_known_faces(self):
        # Load known faces from the known_faces directory
        for filename in os.listdir(self.KNOWN_FACES_DIR):
            filepath = os.path.join(self.KNOWN_FACES_DIR, filename)
            image = cv2.imread(filepath, 0)
            self.known_faces_encodings.append(image)
            self.known_faces_names.append(os.path.splitext(filename)[0])

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

    def store_new_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if not results.detections:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return

        face_count = 0
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            face_image = frame[y:y + h, x:x + w]
            person_name = input(f"Enter the name of person {face_count + 1}: ").strip()
            filename = os.path.join(self.NEW_FACES_DIR, f"{person_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            cv2.imwrite(filename, face_image)
            print(f"New face stored: {filename}")
            face_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def detect_objects(self, frame):
        results = self.object_detection_model.predict(source=frame, conf=0.5, imgsz=320)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            label = self.object_detection_model.names[int(result.cls[0])]
            confidence = result.conf[0].item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def scan_room(self, frame):
        print("Starting room scan. Move the camera to capture the entire room.")
        frames = []

        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 20:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame during room scan.")
                break

            frames.append(frame)
            cv2.imshow("Room Scanning", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Converting frames into room layout...")

        if frames:
            stitched_image = cv2.hconcat(frames[:min(10, len(frames))])  # Take up to 10 frames for demo
            panorama_path = "room_layout.jpg"
            cv2.imwrite(panorama_path, stitched_image)
            print(f"Room layout image saved as {panorama_path}")
        else:
            print("No frames captured for panorama.")

    def process_feed(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            for task in self.running_tasks:
                task(frame)

            cv2.imshow("Live Camera Feed", frame)

            user_command = input("Enter task (recognize, store_face, detect_objects, scan_room, quit): ").strip().lower()

            if user_command == "recognize":
                if "recognize" not in self.running_tasks:
                    self.running_tasks.append(self.recognize_person)
            elif user_command == "store_face":
                self.running_tasks = [self.store_new_face]  # Only store face task active
            elif user_command == "detect_objects":
                if "detect_objects" not in self.running_tasks:
                    self.running_tasks.append(self.detect_objects)
            elif user_command == "scan_room":
                if "scan_room" not in self.running_tasks:
                    self.running_tasks.append(self.scan_room)
            elif user_command == "quit":
                print("Exiting.")
                break
            else:
                print("Unknown command.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    live_feed = LiveCameraFeed()
    live_feed.process_feed()
