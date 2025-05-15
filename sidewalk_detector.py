import torch
from ultralytics import YOLO
import numpy as np
import cv2
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import config  # Make sure this file exists or handle its absence

class SidewalkDetector:
    """
    Unified detector for performing segmentation on both images and videos
    using YOLOv8 segmentation models.
    """
    def __init__(self, model_path=config.SIDEWALK_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        self.class_names = self.model.names if self.model else {}
        self.gpu_info = self._get_gpu_info()
        print(f"YOLO Segmentation Detector initialized on {self.device}.")
        if self.gpu_info:
            print(self.gpu_info)

    def _load_model(self, model_path):
        try:
            model = YOLO(model_path)
            model.to(self.device)
            print(f"Segmentation model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"FATAL: Failed to load YOLO segmentation model from {model_path}\nError: {e}")
            return None

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

    def predict_image(self, image_path: str, confidence: float = 0.25, 
                     selected_classes_idx: Optional[List[int]] = None, 
                     save_output: bool = True) -> Tuple[np.ndarray, Dict, Optional[str]]:
        """
        Process a single image and return segmentation results.
        
        Args:
            image_path: Path to the image file
            confidence: Confidence threshold for detections
            selected_classes_idx: List of class indices to detect, None for all classes
            save_output: Whether to save the annotated image
            
        Returns:
            - annotated_image: Image with visualized detections
            - polygons_by_class: Dictionary of class names to polygon coordinates
            - error_message: Error message if any, None otherwise
        """
        try:
            if not os.path.exists(image_path):
                return None, {}, f"Image not found: {image_path}"
                
            image = cv2.imread(image_path)
            if image is None:
                return None, {}, f"Failed to read image: {image_path}"
                
            annotated_frame, polygons_by_class, error = self.predict_and_get_polygons(
                image, confidence, selected_classes_idx
            )
            
            if save_output and annotated_frame is not None and not error:
                output_path = self._get_output_path(image_path, "images")
                cv2.imwrite(output_path, annotated_frame)
                print(f"Annotated image saved to {output_path}")
                
            return annotated_frame, polygons_by_class, error
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, {}, f"Error processing image: {str(e)}"

    def process_video(self, video_path: str, confidence: float = 0.25,
                     selected_classes_idx: Optional[List[int]] = None,
                     save_output: bool = True, display: bool = False) -> Tuple[str, Optional[str]]:
        """
        Process a video file and apply segmentation to each frame.
        
        Args:
            video_path: Path to the video file
            confidence: Confidence threshold for detections
            selected_classes_idx: List of class indices to detect, None for all classes
            save_output: Whether to save the processed video
            display: Whether to display frames during processing
            
        Returns:
            - output_path: Path to the saved video if save_output is True, else empty string
            - error_message: Error message if any, None otherwise
        """
        try:
            if not os.path.exists(video_path):
                return "", f"Video not found: {video_path}"
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "", f"Failed to open video: {video_path}"
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prepare output video writer if saving is requested
            output_path = ""
            out = None
            if save_output:
                output_path = self._get_output_path(video_path, "videos")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', depending on your system
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            polygon_history = []  # Store polygons for each frame if needed for analysis
            
            print(f"Processing video with {total_frames} frames...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                annotated_frame, polygons_by_class, error = self.predict_and_get_polygons(
                    frame, confidence, selected_classes_idx
                )
                
                if error:
                    print(f"Warning on frame {frame_count}: {error}")
                
                # Optionally store polygon data for later analysis
                polygon_history.append(polygons_by_class)
                
                # Add frame number and processing info
                cv2.putText(
                    annotated_frame, 
                    f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Save to output video
                if out is not None:
                    out.write(annotated_frame)
                
                # Display if requested
                if display:
                    cv2.imshow("Processing Video", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
                
                frame_count += 1
                
                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} FPS)")
            
            # Cleanup
            cap.release()
            if out is not None:
                out.release()
            if display:
                cv2.destroyAllWindows()
                
            print(f"Video processing complete. Processed {frame_count} frames.")
            return output_path, None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return "", f"Error processing video: {str(e)}"

    def predict_and_get_polygons(self, image_source, confidence=0.25, selected_classes_idx=None):
        """
        Performs detection on an image or video frame and extracts segmentation polygons.
        
        Args:
            image_source: Image or frame as numpy array
            confidence: Detection confidence threshold
            selected_classes_idx: List of class indices to detect, None for all classes
            
        Returns:
            - annotated_frame: Image with detections and segmentations plotted
            - polygons_by_class: Dict where keys are class names and values are lists of polygons
                                Each polygon is a list of [x, y] points
            - error_message: String, None if no error
        """
        if not self.model:
            return image_source, {}, "Model not loaded."

        try:
            results = self.model.predict(
                source=image_source,
                conf=confidence,
                classes=selected_classes_idx,
                verbose=False
            )

            if not results or len(results) == 0:
                return image_source, {}, "No results from model."

            result = results[0]  # Assuming single image/frame
            annotated_frame = result.plot()  # Plots bboxes and masks

            polygons_by_class = {name: [] for name in self.class_names.values()}

            if result.masks is not None and hasattr(result.masks, 'xy'):
                for i, mask_coords_np_array in enumerate(result.masks.xy):
                    # Get class ID and confidence
                    if result.boxes and i < len(result.boxes) and result.boxes[i].cls is not None:
                        cls_id = int(result.boxes[i].cls[0])
                        conf = float(result.boxes[i].conf[0]) if hasattr(result.boxes[i], 'conf') else 0.0
                        cls_name = self.class_names.get(cls_id, f"Unknown_cls_{cls_id}")
                        
                        # Convert polygon points to integers and add to results
                        polygon = mask_coords_np_array.astype(np.int32).tolist()
                        
                        # Store polygon with confidence
                        polygons_by_class.setdefault(cls_name, []).append({
                            'points': polygon,
                            'confidence': conf
                        })
                    else:
                        # Fallback if class info isn't directly tied via box index
                        print(f"Warning: Mask found at index {i} but no corresponding box/class_id.")
                        polygon = mask_coords_np_array.astype(np.int32).tolist()
                        polygons_by_class.setdefault("unknown_mask_polygon", []).append({
                            'points': polygon,
                            'confidence': 0.0
                        })
            
            return annotated_frame, polygons_by_class, None

        except Exception as e:
            import traceback
            print(f"Error during segmentation detection: {e}")
            traceback.print_exc()
            return image_source, {}, f"Detection error: {e}"
    
    def _get_output_path(self, input_path: str, media_type: str) -> str:
        """Generate output path for processed media"""
        input_path = Path(input_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create output directory if it doesn't exist
        output_dir = Path(f"output/{media_type}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        if media_type == "images":
            return str(output_dir / f"{input_path.stem}_segmented_{timestamp}{input_path.suffix}")
        else:  # videos
            return str(output_dir / f"{input_path.stem}_segmented_{timestamp}.mp4")

    def process_realtime(self, camera_id: int = 0, confidence: float = 0.25,
                        selected_classes_idx: Optional[List[int]] = None,
                        save_output: bool = False) -> None:
        """
        Process video stream from camera in real-time with segmentation.
        
        Args:
            camera_id: Camera device ID
            confidence: Detection confidence threshold
            selected_classes_idx: List of class indices to detect, None for all classes
            save_output: Whether to save the processed video
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"Failed to open camera ID {camera_id}")
                return
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30  # Assumed fps for webcam
            
            # Prepare output video writer if saving is requested
            out = None
            if save_output:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_dir = Path("output/realtime")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / f"realtime_segmentation_{timestamp}.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                print(f"Recording output to {output_path}")
            
            print("Press 'q' to quit...")
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                frame_count += 1
                process_start = time.time()
                
                annotated_frame, polygons_by_class, error = self.predict_and_get_polygons(
                    frame, confidence, selected_classes_idx
                )
                
                if error:
                    print(f"Warning: {error}")
                
                # Calculate FPS
                process_time = time.time() - process_start
                fps_text = f"FPS: {1/process_time:.2f}" if process_time > 0 else "FPS: N/A"
                
                # Add info to frame
                cv2.putText(
                    annotated_frame, fps_text, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Save to output video if requested
                if out is not None:
                    out.write(annotated_frame)
                
                # Display results
                cv2.imshow("Real-time Segmentation", annotated_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                # Print stats occasionally
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Average processing speed: {avg_fps:.2f} FPS")
            
            # Cleanup
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            import traceback
            print(f"Error in real-time processing: {e}")
            traceback.print_exc()

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = SidewalkDetector()
    

    # Example 2: Process a video file  
    video_path = "./detect_img/Illegal Parking Detection.mp4"
    if os.path.exists(video_path):
        print(f"\nProcessing video: {video_path}")
        output_path, error = detector.process_video(video_path, display=True)
        if error:
            print(f"Error processing video: {error}")
        else:
            print(f"Processed video saved to: {output_path}")
    
