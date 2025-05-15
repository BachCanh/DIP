# parking_app.py
import os
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, StringVar
import json
from datetime import datetime
import time
import threading

from app import YOLODetectionApp
from car_detector import CarDetector
from ui_components import setup_parking_tab, setup_styles, display_image_on_label
from parking_zone_detector import ParkingZoneDetector
from PIL import Image, ImageTk
from sidewalk_detector import SidewalkDetector

class ParkingDetectionApp(YOLODetectionApp):
    def __init__(self, root):
        self.parking_status_var = StringVar(value="Ready. Select video to define zones via sidewalk detection.")
        self.current_zone_file = StringVar(value="auto_sidewalk_zones.json")
        self.alert_threshold = 5
        self.illegal_alerts = [] # For car violations
        self.snapshot_interval = 30
        self.last_snapshot_time = {} # For car violations
        
        self.car_detector_instance = CarDetector() # Explicitly named for clarity
        self.conf_threshold = 0.4

        self.parking_video_path_var = tk.StringVar()
        self.video_size_var = tk.DoubleVar(value=1.0)
        
        self.stop_car_violation_on_first_detected = False # For Stage 2
        self.sidewalk_detection_running = False # Flag for Stage 1 (sidewalk scan)
        self.car_violation_detection_running = False # Flag for Stage 2 (car violations)

        try:
            self.sidewalk_segmentor_instance = SidewalkDetector() # Explicitly named
            if not self.sidewalk_segmentor_instance.model:
                messagebox.showwarning("Sidewalk Model Error", 
                                       "Sidewalk model failed to load. Auto zone generation disabled.")
                self.sidewalk_segmentor_instance = None
        except Exception as e:
            messagebox.showerror("Init Error", f"Failed to init SidewalkDetector: {e}")
            self.sidewalk_segmentor_instance = None

        # Call super().__init__ AFTER setting up attributes that might be used by its methods
        # or by methods it calls (like update_status if YOLODetectionApp defines a self.detector)
        # For now, let's assume YOLODetectionApp's self.detector is separate or handled by Option 1 (hasattr check)
        super().__init__(root) 
        
        self.zone_detector = ParkingZoneDetector()
        self._reorder_tabs()
        self._add_parking_tab()
        self._load_initial_zones()

    def _load_initial_zones(self):
        initial_zone_file = "parking_zones.json" 
        if os.path.exists(os.path.join(self.zone_detector.config_dir, initial_zone_file)):
            if self.zone_detector.load_zones(initial_zone_file):
                self.current_zone_file.set(initial_zone_file)
                self.update_status(f"Loaded existing zones from '{initial_zone_file}'.")
                return True
        # Don't show warning if sidewalk segmentor is available, as that's the primary flow
        if not self.sidewalk_segmentor_instance:
            self.update_status("No pre-existing zones. Sidewalk detector unavailable. Load zones manually.", warning=True)
        else:
            self.update_status("No pre-existing zones. Select video for auto-generation via sidewalk scan.")
        return False

    def _reorder_tabs(self):
        self.notebook = None
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Notebook): self.notebook = child; return
            if isinstance(child, ttk.Frame):
                for gc in child.winfo_children():
                    if isinstance(gc, ttk.Notebook): self.notebook = gc; return
        if not self.notebook: print("CRITICAL: Notebook not found.")

    def _add_parking_tab(self):
        if not self.notebook: self._reorder_tabs()
        if not self.notebook: messagebox.showerror("UI Error", "Notebook missing."); return
        # Ensure setup_parking_tab is aware of the new button commands
        parking_tab = setup_parking_tab(self.notebook, self)
        self.notebook.add(parking_tab, text=" Parking Detection ")

    def _browse_parking_video(self):
        if not self.sidewalk_segmentor_instance: # Check if SidewalkDetector is ready
            messagebox.showerror("Error", "SidewalkDetector is not available. Cannot auto-generate zones.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Video for Sidewalk Zone Generation",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if not file_path:
            return

        self.parking_video_path_var.set(file_path)
        self.update_status(f"Video selected: {os.path.basename(file_path)}. Starting sidewalk scan...")
        
        # Update UI: Disable "Browse", "Start Car Violation"; Enable "Stop Sidewalk Scan"
        if hasattr(self, 'browse_video_btn'): self.browse_video_btn.config(state=tk.DISABLED)
        if hasattr(self, 'parking_detect_btn'): self.parking_detect_btn.config(state=tk.DISABLED)
        if hasattr(self, 'stop_sidewalk_scan_btn'): self.stop_sidewalk_scan_btn.config(state=tk.NORMAL)
        if hasattr(self, 'parking_stop_btn'): self.parking_stop_btn.config(state=tk.DISABLED)


        self.sidewalk_detection_running = True
        threading.Thread(target=self._process_video_for_first_sidewalk, 
                         args=(file_path,), daemon=True).start()

    def _stop_sidewalk_scan_process(self): # Renamed for clarity
        """Connected to a 'Stop Sidewalk Scan' button in UI."""
        self.sidewalk_detection_running = False # Signal the loop to stop
        # The loop itself will call _reset_ui_after_sidewalk_scan upon exiting
        self.update_status("Sidewalk scan stopping...") 
        # Don't immediately change button states here, let the thread finish cleanly.

    def _process_video_for_first_sidewalk(self, video_path):
        if not self.sidewalk_segmentor_instance or not self.sidewalk_segmentor_instance.model:
            self.update_status("SidewalkDetector model not loaded or unavailable.", error=True)
            self.root.after(0, self._reset_ui_after_sidewalk_scan, False) # Schedule UI reset on main thread
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.update_status(f"Error opening video: {video_path}", error=True)
            self.root.after(0, self._reset_ui_after_sidewalk_scan, False)
            return

        frame_num = 0
        sidewalk_polygons_saved_and_loaded = False
        target_sidewalk_class_names = ['sidewalk-and-stair-image', 'sidewalk','stair'] # Case-insensitive check later
        
        # Get source FPS for sleep calculation, if needed, though UI updates dominate
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0: source_fps = 30 # Default

        while self.sidewalk_detection_running: # Loop controlled by the flag
            ret, frame = cap.read()
            if not ret:
                if not sidewalk_polygons_saved_and_loaded: # Only show if no sidewalk was found yet
                    self.update_status("End of video. No sidewalk found during scan.", warning=True)
                break 
            
            frame_num += 1
            loop_start_time = time.time()

            # Update status less frequently to avoid flooding
            if frame_num % 5 == 0 or frame_num == 1 : # Update every 5 frames or on the first frame
                self.update_status(f"Scanning frame {frame_num} for sidewalk...")

            annotated_frame_from_segmentor, polygons_by_class, error = \
                self.sidewalk_segmentor_instance.predict_and_get_polygons(frame.copy(), confidence=0.3) 

            # Display the current frame being processed in the UI
            # Use the annotated frame if available, otherwise the original
            current_display_frame = annotated_frame_from_segmentor if annotated_frame_from_segmentor is not None and annotated_frame_from_segmentor.size > 0 else frame
            if hasattr(self, 'parking_video_label') and self.parking_video_label.winfo_exists():
                 self.root.after(0, self._update_parking_video_display, current_display_frame)


            if error:
                if frame_num % 10 == 0 : # Log error less frequently
                    print(f"Sidewalk detection error (frame {frame_num}): {error}")
                # Continue scanning unless it's a critical error (not handled here yet)
                time.sleep(0.01) # Small delay
                continue
            
            detected_sidewalk_polygons_for_zone = []
            for cls_name, poly_list in polygons_by_class.items():
                if cls_name.lower() in target_sidewalk_class_names and poly_list:
                    for poly_data in poly_list:
                        if poly_data.get('points'): # Ensure 'points' key exists
                            detected_sidewalk_polygons_for_zone.append(poly_data['points'])
            
            if detected_sidewalk_polygons_for_zone: # If any target polygons found
                self.update_status(f"Sidewalk detected in frame {frame_num}! Processing and saving zones...")
                
                parking_zones_data = {"legal": [], "illegal": detected_sidewalk_polygons_for_zone}
                
                config_dir = self.zone_detector.config_dir
                os.makedirs(config_dir, exist_ok=True)
                auto_zone_filename = f"parking_zones.json" # Consistent name
                self.current_zone_file.set(auto_zone_filename) # Update UI entry
                zone_file_path = os.path.join(config_dir, auto_zone_filename)

                try:
                    with open(zone_file_path, 'w') as f:
                        json.dump(parking_zones_data, f, indent=4)
                    self.update_status(f"Zones from sidewalk saved as '{auto_zone_filename}'. Attempting to load...")

                    if self.zone_detector.load_zones(auto_zone_filename):
                        self.update_status(f"Zones from '{auto_zone_filename}' loaded. Sidewalk scan complete.")
                        sidewalk_polygons_saved_and_loaded = True
                        # Keep showing the frame where sidewalk was found
                        self.root.after(0, self._update_parking_video_display, current_display_frame)
                    else:
                        self.update_status(f"Failed to load '{auto_zone_filename}'. Load manually.", error=True)
                except Exception as e:
                    self.update_status(f"Error saving/loading auto-zones: {e}", error=True)
                
                self.sidewalk_detection_running = False # Signal loop to stop
                # The break will happen in the next iteration due to the flag change.
                # Or, we can break immediately:
                break 
            
            # Control processing speed for UI responsiveness, especially if predict_and_get_polygons is fast
            # This sleep is crucial if the processing loop is very fast.
            # It gives Tkinter time to process its event queue (like button presses for stop_sidewalk_scan_btn)
            # Adjust sleep time as needed. 0.02 is 50 FPS, 0.033 is 30 FPS.
            # Subtract processing time to aim for a target display rate.
            frame_proc_time = time.time() - loop_start_time
            sleep_duration = max(0.001, (1.0 / 20) - frame_proc_time) # Target ~20 FPS for this scan display
            time.sleep(sleep_duration)


        cap.release()
        self.sidewalk_detection_running = False # Explicitly set false again after loop
        # Schedule the UI reset to run on the main Tkinter thread
        self.root.after(0, self._reset_ui_after_sidewalk_scan, sidewalk_polygons_saved_and_loaded)

    def _reset_ui_after_sidewalk_scan(self, zones_were_loaded_successfully=False):
        """Resets UI elements after the sidewalk scan process. Called from main thread."""
        if hasattr(self, 'browse_video_btn'): self.browse_video_btn.config(state=tk.NORMAL)
        if hasattr(self, 'stop_sidewalk_scan_btn'): self.stop_sidewalk_scan_btn.config(state=tk.DISABLED)

        if zones_were_loaded_successfully:
            if hasattr(self, 'parking_detect_btn'): self.parking_detect_btn.config(state=tk.NORMAL)
            # Status already set by the processing thread, or could be refined here.
            # self.update_status("Sidewalk scan complete. Zones loaded. Ready for car parking detection.")
        else:
            if hasattr(self, 'parking_detect_btn'): self.parking_detect_btn.config(state=tk.DISABLED)
            # Status already set, e.g., "No sidewalk found" or "Failed to load"
            # self.update_status("Sidewalk scan finished. No new zones loaded. Load zones manually.", warning=True)
        
        # Ensure car violation stop button is disabled if not running
        if not self.car_violation_detection_running:
            if hasattr(self, 'parking_stop_btn'): self.parking_stop_btn.config(state=tk.DISABLED)


    # --- Stage 2: Car Parking Violation Detection ---
    def _start_parking_detection(self): # This is for STAGE 2 (Car Violations)
        video_path = self.parking_video_path_var.get()
        if not video_path:
            messagebox.showwarning("Warning", "No video file specified for parking detection.")
            return
        self.zone_detector.load_zones()
        if not (self.zone_detector.legal_zones or self.zone_detector.illegal_zones):
            messagebox.showwarning("Warning", "No parking zones are loaded. Cannot start car violation detection.")
            return
            
        if hasattr(self, 'parking_detect_btn'): self.parking_detect_btn.config(state=tk.DISABLED)
        if hasattr(self, 'parking_stop_btn'): self.parking_stop_btn.config(state=tk.NORMAL)
        if hasattr(self, 'browse_video_btn'): self.browse_video_btn.config(state=tk.DISABLED)
        if hasattr(self, 'stop_sidewalk_scan_btn'): self.stop_sidewalk_scan_btn.config(state=tk.DISABLED)
        
        if hasattr(self, 'violations_tree'): self.violations_tree.delete(*self.violations_tree.get_children())
        self.illegal_alerts = []
        self.last_snapshot_time = {}
        
        self.update_status("Starting car parking violation detection...")
        self.car_violation_detection_running = True
        threading.Thread(target=self._run_car_violation_detection, daemon=True).start()

    def _run_car_violation_detection(self): # STAGE 2
        first_car_violation_processed_this_run = False
        try:
            video_path = self.parking_video_path_var.get()
            video_cap = cv2.VideoCapture(video_path) # Re-open video for car detection
            if not video_cap.isOpened():
                self.update_status("Error opening video for car detection.", error=True)
                self.root.after(0, self._reset_car_violation_ui)
                return
                
            source_fps = video_cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0: source_fps = 30 
            
            self.zone_detector.persistence_threshold = self.alert_threshold
            
            frame_counter_for_fps = 0
            fps_calc_start_time = time.time()
            processing_fps = 0.0

            while self.car_violation_detection_running:
                ret, frame = video_cap.read()
                if not ret:
                    self.update_status("End of video during car violation detection.", warning=True)
                    break 
                
                current_loop_start_time = time.time()
                
                # Use self.car_detector_instance for car detection
                car_detections_raw = self.car_detector_instance.detect(frame.copy()) 
                
                coco_vehicle_classes = [1, 2, 3, 4] 
                vehicle_detections = [
                    (box, conf, cls_id) for box, conf, cls_id in car_detections_raw 
                    if int(cls_id) in coco_vehicle_classes and conf >= self.conf_threshold
                ]
                
                annotated_frame_cars = frame.copy()
                # Pass the car_detector_instance to _draw_boxes_and_labels
                annotated_frame_cars = self._draw_boxes_and_labels(annotated_frame_cars, vehicle_detections, 
                                                                   detector_instance=self.car_detector_instance)
                
                frame_with_zones, illegal_events = self.zone_detector.check_illegal_parking(
                    annotated_frame_cars, vehicle_detections
                )
                
                self._process_illegal_events(illegal_events, frame) # Original frame for snapshot

                if self.stop_car_violation_on_first_detected and illegal_events:
                    if not first_car_violation_processed_this_run:
                        self.update_status("First car parking violation detected. Stopping car detection.", warning=True)
                        first_car_violation_processed_this_run = True
                    self.car_violation_detection_running = False 
                
                # FPS Calculation
                frame_counter_for_fps += 1
                if (time.time() - fps_calc_start_time) >= 1.0:
                    elapsed_time = (time.time() - fps_calc_start_time)
                    processing_fps = frame_counter_for_fps / elapsed_time if elapsed_time > 0 else 0.0
                    frame_counter_for_fps = 0
                    fps_calc_start_time = time.time()

                self.root.after(0, self._update_parking_ui, frame_with_zones, processing_fps)
                
                if not self.car_violation_detection_running: break

                loop_proc_time = time.time() - current_loop_start_time
                sleep_duration = max(0.001, (1.0 / source_fps) - loop_proc_time)
                time.sleep(sleep_duration)
                
            video_cap.release()
        except Exception as e:
            self.update_status(f"Error in car violation detection: {e}", error=True)
            import traceback; traceback.print_exc()
        finally:
            # Schedule UI reset on the main Tkinter thread
            self.root.after(0, self._reset_car_violation_ui)


    def _stop_car_violation_detection_process(self):
        self.car_violation_detection_running = False
        # The loop will see the flag and call _reset_car_violation_ui
        self.update_status("Car parking violation detection stopping...")

    def _reset_car_violation_ui(self):
        """Resets UI buttons after car violation detection stops/finishes."""
        if hasattr(self, 'parking_detect_btn'): self.parking_detect_btn.config(state=tk.NORMAL)
        if hasattr(self, 'parking_stop_btn'): self.parking_stop_btn.config(state=tk.DISABLED)
        if hasattr(self, 'browse_video_btn'): self.browse_video_btn.config(state=tk.NORMAL)
        # Status already updated by the processing thread or stop action
        # self.update_status("Car parking violation detection stopped.")


    def _update_parking_video_display(self, frame):
        if frame is None or not hasattr(self, 'parking_video_label') or not self.parking_video_label.winfo_exists():
            return
        
        # Make a copy to avoid modifying the frame if it's used elsewhere
        display_frame = frame.copy()
        self.current_parking_frame = display_frame # Keep reference to the frame being displayed

        label_width = self.parking_video_label.winfo_width()
        label_height = self.parking_video_label.winfo_height()

        if label_width < 20 or label_height < 20: # Min sensible size for label
            # Try to get parent size if label is not ready or too small
            parent_width = self.parking_video_label.master.winfo_width()
            parent_height = self.parking_video_label.master.winfo_height()
            if parent_width > 20 and parent_height > 20:
                label_width, label_height = parent_width, parent_height
            else: # Fallback to a default if parent also not ready
                label_width, label_height = 640, 480 

        fh, fw = display_frame.shape[:2]
        if fw == 0 or fh == 0: return

        # Calculate scale to fit frame into label while maintaining aspect ratio
        scale = min(label_width / fw, label_height / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        if nw <= 0 or nh <= 0: return # Avoid invalid resize dimensions
        
        try:
            resized = cv2.resize(display_frame, (nw, nh), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.parking_video_label.config(image=photo)
            self.parking_video_label.image = photo # Keep reference
        except Exception as e:
            print(f"Error updating video display: {e}")


    def _draw_boxes_and_labels(self, frame, detections, detector_instance):
        # Ensure detector_instance and its model/names are valid
        if not detector_instance or not hasattr(detector_instance, 'model') or not hasattr(detector_instance.model, 'names'):
            class_names = {}
        else:
            class_names = detector_instance.model.names
        
        for box, conf, cls_id_tensor in detections:
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id_tensor)
            name = class_names.get(cls_id, f"ID:{cls_id}")
            label = f"{name} {conf:.2f}"
            # Differentiate color based on detector type if needed, or use a consistent color
            color = (0, 255, 0) # Green for car detections
            # if detector_instance == self.sidewalk_segmentor_instance: color = (255, 0, 0) # Red for sidewalk
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) # Smaller font
        return frame

    def _process_illegal_events(self, events, original_frame_for_snapshot):
        current_time = datetime.now()
        center_tolerance = 50 # pixels
        
        for event in events: # event should be {'bbox': [...], 'center': (x,y)}
            bbox = event.get('bbox')
            center = event.get('center')
            if bbox is None or center is None: 
                print(f"Warning: Malformed event in _process_illegal_events: {event}")
                continue

            matched_alert = None
            for alert in self.illegal_alerts:
                # Basic distance check for matching
                dist_sq = (center[0] - alert['center'][0])**2 + (center[1] - alert['center'][1])**2
                if dist_sq < center_tolerance**2:
                    matched_alert = alert
                    break
            
            if matched_alert: # Update existing
                matched_alert['last_seen'] = current_time
                matched_alert['center'] = center 
                matched_alert['bbox'] = bbox
                duration = (current_time - matched_alert['first_seen']).total_seconds()
                matched_alert['duration'] = duration
                
                if hasattr(self, 'violations_tree') and matched_alert.get('tree_id') and \
                   self.violations_tree.winfo_exists() and self.violations_tree.exists(matched_alert['tree_id']):
                    try:
                        self.violations_tree.item(matched_alert['tree_id'], values=(
                            matched_alert['first_seen'].strftime("%H:%M:%S"), 
                            f"X:{int(center[0])},Y:{int(center[1])}", 
                            f"{int(duration)}s"))
                    except tk.TclError as e:
                        print(f"Treeview update error: {e}")
                
                alert_id = matched_alert['id']
                if alert_id not in self.last_snapshot_time or \
                   (current_time - self.last_snapshot_time[alert_id]).total_seconds() >= self.snapshot_interval:
                    self._take_violation_snapshot(original_frame_for_snapshot, bbox, alert_id, current_time)
                    self.last_snapshot_time[alert_id] = current_time
            else: # New violation
                viol_id = f"car_viol_{current_time.strftime('%Y%m%d_%H%M%S_%f')}" # More unique
                tree_item_id = viol_id # Use same ID for tree item for consistency
                
                if hasattr(self, 'violations_tree') and self.violations_tree.winfo_exists():
                    try: 
                        self.violations_tree.insert("", "end", iid=tree_item_id, values=(
                            current_time.strftime("%H:%M:%S"), 
                            f"X:{int(center[0])},Y:{int(center[1])}", "0s"))
                    except tk.TclError as e: # If iid somehow already exists
                        print(f"Treeview insert error (iid conflict?): {e}")
                        tree_item_id = None # Don't use tree_id if insert failed
                else: 
                    tree_item_id = None

                new_alert_data = {'id': viol_id, 'tree_id': tree_item_id, 
                                  'first_seen': current_time, 'last_seen': current_time, 
                                  'center': center, 'bbox': bbox, 'duration': 0.0}
                self.illegal_alerts.append(new_alert_data)
                self._take_violation_snapshot(original_frame_for_snapshot, bbox, viol_id, current_time)
                self.last_snapshot_time[viol_id] = current_time
        
        self._clean_stale_violations(current_time)

    def _clean_stale_violations(self, current_time):
        stale_threshold_seconds = 10 
        
        # Identify stale alerts without modifying list during iteration
        stale_alerts_ids_to_remove = {
            alert['id'] for alert in self.illegal_alerts 
            if (current_time - alert['last_seen']).total_seconds() > stale_threshold_seconds
        }

        if not stale_alerts_ids_to_remove:
            return
        
        # Rebuild self.illegal_alerts excluding stale ones
        cleaned_alerts = []
        removed_count = 0
        for alert in self.illegal_alerts:
            if alert['id'] in stale_alerts_ids_to_remove:
                if hasattr(self, 'violations_tree') and alert.get('tree_id') and \
                   self.violations_tree.winfo_exists() and self.violations_tree.exists(alert['tree_id']):
                    try:
                        self.violations_tree.delete(alert['tree_id'])
                    except tk.TclError as e:
                        print(f"Error deleting from treeview: {e}")
                
                if alert['id'] in self.last_snapshot_time:
                    del self.last_snapshot_time[alert['id']]
                removed_count +=1
            else:
                cleaned_alerts.append(alert)
        
        self.illegal_alerts = cleaned_alerts
        
        if removed_count > 0:
            self.update_status(f"Cleaned {removed_count} stale car violation(s).")


    def _take_violation_snapshot(self, frame, bbox, violation_id, timestamp):
        try:
            snap_dir = os.path.join("output", "car_snapshots") # Changed dir name slightly
            os.makedirs(snap_dir, exist_ok=True)
            x1,y1,x2,y2 = map(int, bbox); margin=20; h,w=frame.shape[:2]
            # Ensure coordinates are valid after adding margin
            sx1,sy1 = max(0, x1 - margin), max(0, y1 - margin)
            sx2,sy2 = min(w, x2 + margin), min(h, y2 + margin)
            
            # Ensure the slice is valid
            if sx1 >= sx2 or sy1 >= sy2: # If margin makes it invalid, use original bbox
                 snapshot_region = frame[y1:y2, x1:x2].copy()
            else:
                 snapshot_region = frame[sy1:sy2, sx1:sx2].copy()

            if snapshot_region.size == 0: 
                print(f"Warning: Snapshot region for {violation_id} is empty.")
                return
            
            filename = f"car_snap_{violation_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(snap_dir, filename)
            if not cv2.imwrite(filepath, snapshot_region):
                print(f"Warning: Failed to write snapshot {filepath}")

        except Exception as e: 
            self.update_status(f"Snapshot error for {violation_id}: {e}", error=True)
            print(f"Full snapshot error: {e}, {type(e)}")

    def _update_parking_ui(self, frame_to_display, current_fps):
        # This is for Stage 2 (Car Violation) UI updates
        if frame_to_display is None: return

        display_overlay_frame = frame_to_display.copy()
        font_scale=0.5; thickness=1 # Smaller font for less clutter
        
        # FPS display
        cv2.putText(display_overlay_frame, f"FPS: {current_fps:.1f}", (10,15), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
        # Violation count display
        cv2.putText(display_overlay_frame, f"Car Violations: {len(self.illegal_alerts)}", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)
        
        # Update the main video label (runs on main thread via self.root.after)
        self._update_parking_video_display(display_overlay_frame)
        
        # Update status bar less frequently or only on significant events
        # For example, update status bar when a new violation is added or FPS changes significantly.
        # self.update_status(f"Car detection running | FPS: {current_fps:.1f} | Violations: {len(self.illegal_alerts)}")


    def _save_parking_violations(self): # For car violations (Stage 2)
        if not self.illegal_alerts: 
            messagebox.showinfo("Save Car Violations", "No active car parking violations to save.")
            return
        try:
            output_dir = os.path.join("output","car_violation_reports")
            os.makedirs(output_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"car_parking_violations_report_{timestamp_str}.json"
            filepath = os.path.join(output_dir, report_filename)
            
            report_violation_data = []
            for alert in self.illegal_alerts:
                report_violation_data.append({
                    "violation_id": alert['id'],
                    "center_coordinates": [int(c) for c in alert['center']],
                    "bounding_box": [int(b) for b in alert['bbox']],
                    "first_detected_at": alert['first_seen'].isoformat(),
                    "last_detected_at": alert['last_seen'].isoformat(),
                    "duration_seconds": round(alert['duration'], 2)
                })
            
            report_content = {
                "report_generated_at": datetime.now().isoformat(),
                "video_source": os.path.basename(self.parking_video_path_var.get()) if self.parking_video_path_var.get() else "N/A",
                "zone_configuration_file": self.current_zone_file.get(),
                "total_violations_in_report": len(report_violation_data),
                "violations_details": report_violation_data
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_content, f, indent=4)
            
            self.update_status(f"Saved {len(report_violation_data)} car violations to '{report_filename}'")
            messagebox.showinfo("Save Success", 
                                f"Saved {len(report_violation_data)} car violations to:\n{filepath}")
        except Exception as e:
            self.update_status(f"Error saving car violations report: {str(e)}", error=True)
            messagebox.showerror("Save Error", f"Failed to save car violations report: {str(e)}")

    def update_status(self, message, clear_after=None, error=False, warning=False):
        if hasattr(self, 'parking_status_var'):
            prefix = "ERROR: " if error else "WARNING: " if warning else ""
            self.parking_status_var.set(prefix + str(message))
            if self.root and self.root.winfo_exists(): self.root.update_idletasks()
            if clear_after: self.root.after(clear_after, lambda: self.parking_status_var.set("Ready."))
        else:
            print(f"Status (UI not ready): {message}")

    def _stop_parking_detection(self): # General stop, might need to be more specific
        # This method is usually connected to a general "Stop Detection" button in ui_components
        # We need to decide if it stops sidewalk scan or car violation scan.
        # For now, let's assume it tries to stop car violation scan primarily.
        if self.car_violation_detection_running:
            self._stop_car_violation_detection_process()
        elif self.sidewalk_detection_running: # If sidewalk scan is somehow still primary
            self._stop_sidewalk_scan_process()
        else:
            self.update_status("Nothing to stop currently.")

    def _resize_video_feed(self, value):
        """Resize the video feed based on the scale value"""
        # Convert scale value to percentage for display
        scale_factor = self.video_size_var.get()
        percentage = int(scale_factor * 100)
        self.scale_label.config(text=f"{percentage}%")
        
        # If we have a current frame being displayed
        if hasattr(self, 'current_parking_frame') and self.current_parking_frame is not None:
            # Resize the frame and update display
            self._update_parking_video_display(self.current_parking_frame, scale_factor=scale_factor)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Parking System: Sidewalk Zones -> Car Violations")
    try:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1280, sw-100)}x{min(820, sh-100)}")
    except tk.TclError: root.geometry("1280x800") # Fallback
    
    setup_styles(root)
    app = ParkingDetectionApp(root)
    root.mainloop()