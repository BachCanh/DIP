# video_processor.py
import threading
import time
import cv2
import config

class VideoProcessor(threading.Thread):
    def __init__(self, video_path, detector, confidence, selected_classes, gui_update_callback, completion_callback, stop_callback):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.detector = detector
        self.confidence = confidence
        self.selected_classes = selected_classes
        self.gui_update_callback = gui_update_callback
        self.completion_callback = completion_callback
        self.stop_callback = stop_callback
        self.running = True
        self.total_frames = 0
        self.accumulated_counts = {name: 0 for name in detector.class_names.values()} if detector else {}

    def stop(self):
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {self.video_path}")
            self.running = False
            # Maybe add an error callback?
            self.stop_callback(error=f"Cannot open video file: {self.video_path}")
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video

            frame_count += 1
            loop_start_time = time.time()

            annotated_frame, frame_detections = self.detector.predict_frame(
                frame, self.confidence, self.selected_classes
            )

            # Accumulate counts
            for class_name, count in frame_detections.items():
                if class_name in self.accumulated_counts:
                    self.accumulated_counts[class_name] += count

            # Calculate FPS for display
            processing_time = time.time() - loop_start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0

            # Schedule GUI update
            self.gui_update_callback(annotated_frame, frame_detections, frame_count, self.total_frames, fps)

            # time.sleep(0.005) # Optional small delay

        cap.release()

        # Call appropriate callback based on how the loop ended
        if not self.running: # Stopped manually
            self.stop_callback(results=self.accumulated_counts)
        else: # Finished naturally
            self.completion_callback(results=self.accumulated_counts)