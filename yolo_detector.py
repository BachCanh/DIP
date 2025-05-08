# yolo_detector.py
import torch
from ultralytics import YOLO
import config

class YOLODetector:
    def __init__(self, model_path=config.MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        self.class_names = self.model.names if self.model else {}
        self.gpu_info = self._get_gpu_info()
        print(f"YOLO Detector initialized on {self.device}.")
        if self.gpu_info:
            print(self.gpu_info)

    def _load_model(self, model_path):
        try:
            model = YOLO(model_path)
            model.to(self.device)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"FATAL: Failed to load YOLO model from {model_path}\nError: {e}")
            # In a real app, you might raise this exception or handle it differently
            return None # Indicate failure

    def _get_gpu_info(self):
        if self.device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return f"GPU detected: {gpu_name} ({gpu_memory:.2f} GB)"
            except Exception as e:
                print(f"Warning: Could not get GPU details: {e}")
                return "GPU detected (details unavailable)"
        return "No GPU detected. Using CPU."

    def predict_image(self, image_source, confidence, selected_classes_idx):
        """Performs detection on a single image."""
        if not self.model:
            return None, None, "Model not loaded."

        try:
            results = self.model.predict(
                source=image_source,
                conf=confidence,
                classes=selected_classes_idx,
                verbose=False # Less console spam
            )

            if not results or len(results) == 0 or results[0].boxes is None:
                return image_source, {}, "No objects detected." # Return original image and empty counts

            annotated_frame = results[0].plot()
            detection_counts = {name: 0 for name in self.class_names.values()}
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"Unknown({cls_id})")
                if cls_name in detection_counts:
                    detection_counts[cls_name] += 1

            return annotated_frame, detection_counts, None # Return annotated frame, counts, no error

        except Exception as e:
            print(f"Error during image detection: {e}")
            return image_source, {}, f"Detection error: {e}" # Return original, empty counts, error message
    # Add this method to your YOLODetector class

    def detect(self, frame):
        """
        Perform YOLO detection on a single frame.
        Returns a list of (bbox, confidence, class_id)
        """
        if not self.model:
            return []

        results = self.model.predict(frame, verbose=False)
        detections = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append(([x1, y1, x2, y2], conf, cls_id))

        return detections


    def predict_frame(self, frame, confidence, selected_classes_idx):
        """Performs detection on a single video frame."""
        # Very similar to predict_image, could be merged if needed
        if not self.model:
            return frame, {} # Return original frame, empty counts if no model

        try:
            results = self.model.predict(
                source=frame,
                conf=confidence,
                classes=selected_classes_idx,
                verbose=False
            )

            if not results or len(results) == 0 or results[0].boxes is None:
                return frame, {} # No detections

            annotated_frame = results[0].plot()
            frame_detections = {name: 0 for name in self.class_names.values()}
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"Unknown({cls_id})")
                if cls_name in frame_detections:
                    frame_detections[cls_name] += 1

            return annotated_frame, frame_detections

        except Exception as e:
            print(f"Error during frame prediction: {e}")
            return frame, {} # Return original on error