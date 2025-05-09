# parking_app.py
import os
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, StringVar
import json
from datetime import datetime
import time
import threading

# Import base components
from app import YOLODetectionApp
import config
from yolo_detector import YOLODetector
from parking_zone_detector import ParkingZoneDetector
from ui_components import setup_styles, create_banner, display_image_on_label, create_status_bar
from video_processor import VideoProcessor
from zone_editor import ZoneEditorApp


class ParkingDetectionApp(YOLODetectionApp):
    def __init__(self, root):
        # Initialize parking-specific variables first
        self.parking_status_var = StringVar(value="Ready for parking detection")
        self.current_zone_file = StringVar(value="parking_zones.json")
        self.alert_threshold = 5  # Frames to confirm illegal parking
        self.illegal_alerts = []  # Track illegal parking alerts
        self.snapshot_interval = 30  # Seconds between snapshots of violations
        self.last_snapshot_time = {}  # Track last snapshot time per violation
        self.detector = YOLODetector()
        self.is_processing_paused_for_editor = False # Add this flag
        # Now initialize base detection app
        super().__init__(root)
        
        # Add parking zone detector
        self.zone_detector = ParkingZoneDetector()
        
        # Adjust the tab order to make Parking tab appear before Settings
        self._reorder_tabs()
        
        # Add Parking tab to notebook
        self._add_parking_tab()
        
        # Try to load default zones
        if os.path.exists(os.path.join(self.zone_detector.config_dir, "parking_zones.json")):
            self.zone_detector.load_zones("parking_zones.json")
            self.update_status(f"Loaded default parking zones")
    
    def _reorder_tabs(self):
        """Reorder tabs to put Parking tab before Settings"""
        # Get reference to the notebook
        self.notebook = None
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.Notebook):
                        self.notebook = grandchild
                        break
        
        if not self.notebook:
            messagebox.showerror("Error", "Could not find notebook widget")
            return
    
    def _add_parking_tab(self):
        """Add a new tab for parking violation detection"""
        if not hasattr(self, 'notebook') or not self.notebook:
            messagebox.showerror("Error", "Could not find notebook widget")
            return
            
        # Create the parking tab
        parking_tab = self._setup_parking_tab(self.notebook)
        
        # Insert the parking tab before the last tab (assuming Settings is last)
        tab_count = self.notebook.index('end')
        if tab_count > 0:
            # Add the tab at the position before the last tab
            self.notebook.insert(tab_count - 1, parking_tab, text=" Parking Detection ")
        else:
            # If no tabs, just add it
            self.notebook.add(parking_tab, text=" Parking Detection ")
        
    def _setup_parking_tab(self, parent):
        """Setup the parking detection tab UI"""
        # In _setup_parking_tab method
        tab = ttk.Frame(parent, padding="5 5 5 5", style="Main.TFrame")
        tab.columnconfigure(0, weight=1) 
        tab.rowconfigure(3, weight=1)   
        
        # File frame
        ff = ttk.LabelFrame(tab, text="Parking Detection Source", padding="10 10")
        ff.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))
        ff.columnconfigure(1, weight=1)
        
        # Use the same source controls as video tab
        ttk.Label(ff, text="Select Video:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
        self.parking_video_path_var = StringVar()
        ttk.Entry(ff, textvariable=self.parking_video_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(ff, text="Browse...", command=self._browse_parking_video).grid(row=0, column=2, padx=5, pady=5)
        
        # Zone configuration frame
        zf = ttk.LabelFrame(tab, text="Parking Zone Configuration", padding="10 10")
        zf.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 10))
        zf.columnconfigure(1, weight=1)
        
        ttk.Label(zf, text="Zone File:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
        ttk.Entry(zf, textvariable=self.current_zone_file, width=40).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(zf, text="Browse...", command=self._browse_zone_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(zf, text="Load Zones", command=self._load_parking_zones).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(zf, text="Edit Zones", command=self._open_zone_editor).grid(row=0, column=4, padx=5, pady=5)
        
        # Control frame
        cf = ttk.Frame(tab, padding="5 0", style="Main.TFrame")
        cf.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 10))
        cf.columnconfigure(3, weight=1)
        
        self.parking_detect_btn = ttk.Button(cf, text="Start Detection", command=self._start_parking_detection)
        self.parking_detect_btn.grid(row=0, column=0, padx=(0, 5), pady=5)
        
        self.parking_stop_btn = ttk.Button(cf, text="Stop Detection", command=self._stop_parking_detection, state=tk.DISABLED)
        self.parking_stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.parking_save_btn = ttk.Button(cf, text="Save Violations", command=self._save_parking_violations, state=tk.DISABLED)
        self.parking_save_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Add display area with split view
        display_frame = ttk.Frame(tab, style="Main.TFrame")
        display_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Left side - Video display
        video_display = ttk.LabelFrame(display_frame, text="Video Feed", padding="5 5")
        video_display.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        
        self.parking_video_label = ttk.Label(video_display)
        self.parking_video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Violations list
        violations_frame = ttk.LabelFrame(display_frame, text="Detected Violations", padding="5 5")
        violations_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        
        # Create a Treeview to display violations
        self.violations_tree = ttk.Treeview(violations_frame, columns=("time", "location", "duration"), show="headings")
        self.violations_tree.heading("time", text="Time")
        self.violations_tree.heading("location", text="Location")
        self.violations_tree.heading("duration", text="Duration")
        self.violations_tree.column("time", width=100)
        self.violations_tree.column("location", width=150)
        self.violations_tree.column("duration", width=100)
        
        scrollbar = ttk.Scrollbar(violations_frame, orient="vertical", command=self.violations_tree.yview)
        self.violations_tree.configure(yscrollcommand=scrollbar.set)
        
        self.violations_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        status_frame = ttk.Frame(tab, style="Main.TFrame")
        status_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=(0, 5))
        
        status_label = ttk.Label(status_frame, textvariable=self.parking_status_var, style="Status.TLabel")
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        return tab
    
    def _browse_parking_video(self):
        """Browse for a video file to use for parking detection"""
        file_path = filedialog.askopenfilename(
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if file_path:
            self.parking_video_path_var.set(file_path)
            self.update_status(f"Selected parking video: {os.path.basename(file_path)}")
            
            # Auto-extract first frame and open zone editor
            self._extract_first_frame_and_edit_zones(file_path)
    
    def _extract_first_frame_and_edit_zones(self, video_path):
        """Extract the first frame from the video file and open the zone editor"""
        try:
            # Create temporary directory for frames if it doesn't exist
            frames_dir = os.path.join("temp", "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Open video and read first frame
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                self.update_status("Error: Could not open video file")
                return False
                
            ret, frame = video.read()
            if not ret:
                self.update_status("Error: Could not read first frame from video")
                return False
                
            # Save the first frame
            frame_path = os.path.join(frames_dir, "first_frame.jpg")
            cv2.imwrite(frame_path, frame)
            video.release()
            
            # Ask user if they want to define zones now
            if messagebox.askyesno("Zone Setup", "Would you like to define parking zones using the first frame of the video?"):
                # Create custom zone editor with the first frame pre-loaded
                self._open_zone_editor_with_image(frame_path)
                return True
            
            return False
            
        except Exception as e:
            self.update_status(f"Error extracting first frame: {str(e)}")
            return False
    
    def _open_zone_editor_with_image(self, image_path):
        """Open the zone editor with a specific image preloaded, and disable the main window"""
        # Create a new top-level window
        editor_window = tk.Toplevel(self.root)
        editor_window.title("Parking Zone Editor")
        
        # Make sure it has proper dimensions
        editor_window.geometry("1024x768")
        
        # Initialize the zone editor
        editor_app = ZoneEditorApp(editor_window)

        # Keep the main window state preserved by disabling interaction but NOT changing attributes
        # that would cause layout recalculation
        self.root.attributes('-disabled', True)
        
        # Make sure this window stays on top of the main window
        editor_window.transient(self.root)
        editor_window.grab_set()
        
        # Store the original size of the main window
        original_geometry = self.root.geometry()

        def on_close():
            # Re-enable the main window
            self.root.attributes('-disabled', False)
            
            # Restore the original geometry if needed
            self.root.geometry(original_geometry)
            
            # Release the grab and destroy the editor window
            editor_window.grab_release()
            editor_window.destroy()
            
            # Force an update of the main window layout
            self.root.update_idletasks()

        editor_window.protocol("WM_DELETE_WINDOW", on_close)

        # Schedule loading the image after the editor window is fully created
        editor_window.after(100, lambda: self._load_image_in_editor(editor_app, image_path))
    
    def _load_image_in_editor(self, editor_app, image_path):
        """Load the specified image into the zone editor"""
        editor_app.image_path = image_path
        editor_app.current_image = cv2.imread(image_path)
        if editor_app.current_image is not None:
            editor_app._update_canvas()
            
            # Display messagebox with the editor window as parent
            # This ensures proper window stacking order
            messagebox.showinfo(
                "Zone Definition",
                "Define parking zones by clicking points on the image.\n\n"
                "1. Select 'Legal Zone' or 'Illegal Zone'\n"
                "2. Click points to create a polygon\n"
                "3. Click 'Complete Zone' when finished\n"
                "4. Save the zones when done",
                parent=editor_app.root  # Set the parent explicitly
            )
    
    def _browse_zone_file(self):
        """Browse for a parking zone definition file"""
        file_path = filedialog.askopenfilename(
            initialdir=self.zone_detector.config_dir,
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if file_path:
            self.current_zone_file.set(os.path.basename(file_path))
            self.update_status(f"Selected zone file: {os.path.basename(file_path)}")
    
    # In parking_app.py (ParkingDetectionApp class)

    def _load_parking_zones(self):
        """Load parking zones from selected file."""
        ui_filename_input = self.current_zone_file.get() # Filename from the UI entry
        if not ui_filename_input:
            messagebox.showwarning("Load Zones", "Please enter or select a zone filename.")
            self.update_status("Zone loading cancelled: No filename provided.", warning=True)
            return

        # Ensure it has a .json extension
        if not ui_filename_input.endswith('.json'):
            ui_filename_input += '.json'

       
        filename_to_load = os.path.basename(ui_filename_input)
        print(f"Filename being passed to zone_detector.load_zones(): '{filename_to_load}'")


        if self.zone_detector.load_zones(filename_to_load): # Pass the (potentially basename) filename
            self.update_status(f"Successfully loaded zones from '{filename_to_load}' (in config dir).")
            messagebox.showinfo("Load Zones Success", f"Zones loaded successfully from '{filename_to_load}'.")
        else:
            # The actual full path ParkingZoneDetector tried to load will be useful here.
            # We can't directly get it from the return False, so we construct it for the message.
            expected_full_path = os.path.join(self.zone_detector.config_dir, filename_to_load)
            error_msg = f"Could not load zones from '{filename_to_load}'.\n" \
                        f"Expected file at: {expected_full_path}\n\n" \
                        f"Please check:\n" \
                        f"1. The file exists in the directory: '{self.zone_detector.config_dir}'.\n" \
                        f"2. The file is a valid JSON format.\n" \
                        f"3. The filename in the UI is correct."
            self.update_status(f"Failed to load zones from '{filename_to_load}'. See error dialog.", error=True)
            messagebox.showerror("Load Zones Error", error_msg)

    
    def _open_zone_editor(self):
        """Open the zone editor with the current video frame"""
        video_path = self.parking_video_path_var.get()
        if not video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return

        # Extract the current frame and open the editor
        self._extract_current_frame_and_open_zone_editor(video_path)
    def _extract_current_frame_and_open_zone_editor(self, video_path):
        """Extract the current frame from the video and open the zone editor dialog"""
        try:
            frames_dir = os.path.join("temp", "frames")
            os.makedirs(frames_dir, exist_ok=True)

            # Open video and read the current frame (or first frame for simplicity)
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                self.update_status("Error: Could not open video file")
                return

            # If you want the current frame, you may want to track the frame number elsewhere.
            # For now, just grab the first frame:
            ret, frame = video.read()
            if not ret:
                self.update_status("Error: Could not read frame from video")
                return

            frame_path = os.path.join(frames_dir, "current_frame.jpg")
            cv2.imwrite(frame_path, frame)
            video.release()

            # Now open the zone editor with this frame
            self._open_zone_editor_with_image(frame_path)

        except Exception as e:
            self.update_status(f"Error extracting frame: {str(e)}")
        
    def _start_parking_detection(self):
        """Start parking violation detection"""
        # Check if video file is selected
        video_path = self.parking_video_path_var.get()
        if not video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return
            
        # Check if zones are loaded
        if not self.zone_detector.legal_zones and not self.zone_detector.illegal_zones:
            messagebox.showwarning("Warning", "No parking zones defined. Please load or create zones first.")
            return
            
        # Update UI state
        self.parking_detect_btn.config(state=tk.DISABLED)
        self.parking_stop_btn.config(state=tk.NORMAL)
        self.parking_save_btn.config(state=tk.NORMAL)
        
        # Clear previous violations
        self.violations_tree.delete(*self.violations_tree.get_children())
        self.illegal_alerts = []
        
        # Start video processing in a new thread
        self.update_status("Starting parking violation detection...")
        self.detection_running = True
        
        threading.Thread(target=self._run_parking_detection, daemon=True).start()
    
    def _run_parking_detection(self):
        """Run the parking violation detection process"""
        try:
            # Open video
            video_path = self.parking_video_path_var.get()
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                self.update_status("Error: Could not open video")
                messagebox.showerror("Error", "Could not open video file")
                self._reset_parking_detection_ui()
                return
                
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_count = 0
            processing_fps = 0
            last_time = time.time()
            
            # Configure zone detector persistence threshold
            self.zone_detector.persistence_threshold = self.alert_threshold
            
            while self.detection_running:
                # Read frame
                ret, frame = video.read()
                
                if not ret:
                    # End of video, loop back to beginning
                    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process frame with detector
                detections = self.detector.detect(frame)
                
                # Filter for vehicle classes (car, truck, bus)
                vehicle_classes = [1, 2, 3, 5, 7] # COCO class IDs
                vehicle_detections = [
                    (box, conf, cls) for box, conf, cls in detections 
                    if int(cls) in vehicle_classes and conf > self.conf_threshold
                ]
                
                # Check for illegal parking
                annotated_frame, illegal_events = self.zone_detector.check_illegal_parking(
                    frame, vehicle_detections
                )
                
                # Process illegal parking events
                self._process_illegal_events(illegal_events, annotated_frame)
                
                # Update UI
                frame_count += 1
                now = time.time()
                if now - last_time >= 1.0:
                    processing_fps = frame_count / (now - last_time)
                    frame_count = 0
                    last_time = now
                
                self._update_parking_ui(annotated_frame, processing_fps)
                time.sleep(0.01)  # Small delay to prevent UI freezing
                
            video.release()
            
        except Exception as e:
            self.update_status(f"Error during detection: {str(e)}")
            messagebox.showerror("Error", f"Detection error: {str(e)}")
        finally:
            self._reset_parking_detection_ui()
    
    def _process_illegal_events(self, events, frame):
        """Process detected illegal parking events"""
        current_time = datetime.now()
        
        # Tolerance for matching vehicle centers (in pixels)
        center_tolerance = 50  # Adjust based on video resolution and vehicle size
        
        for event in events:
            # Create a unique ID for this violation
            bbox = event['bbox']
            center = event['center']
            location_id = f"{center[0]}_{center[1]}"  # Keep for compatibility
            
            # Check if this is a new violation
            is_new = True
            for alert in self.illegal_alerts:
                # Calculate Euclidean distance between current center and alert center
                alert_center = alert['center']
                distance = ((center[0] - alert_center[0])**2 + (center[1] - alert_center[1])**2)**0.5
                
                if distance < center_tolerance:
                    # Update existing alert
                    is_new = False
                    alert['last_seen'] = current_time
                    alert['center'] = center  # Update center to latest position
                    alert['bbox'] = bbox      # Update bbox to latest detection
                    
                    # Calculate duration
                    duration = (current_time - alert['first_seen']).total_seconds()
                    alert['duration'] = duration
                    
                    # Update treeview
                    self.violations_tree.item(
                        alert['tree_id'],
                        values=(
                            alert['first_seen'].strftime("%H:%M:%S"),
                            f"X: {center[0]}, Y: {center[1]}",
                            f"{int(duration)} sec"
                        )
                    )
                    
                    # Take periodic snapshots
                    if alert['id'] not in self.last_snapshot_time or \
                    (current_time - self.last_snapshot_time[alert['id']]).total_seconds() > self.snapshot_interval:
                        self._take_violation_snapshot(frame, bbox, alert['id'], current_time)
                        self.last_snapshot_time[alert['id']] = current_time
                    
                    break
            
            # Add new violation
            if is_new:
               
                tree_id = self.violations_tree.insert(
                    "", "end", 
                    values=(
                        current_time.strftime("%H:%M:%S"),
                        f"X: {center[0]}, Y: {center[1]}",
                        "0 sec"
                    )
                )
                
                self.illegal_alerts.append({
                    'id': location_id,
                    'tree_id': tree_id,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'center': center,
                    'bbox': bbox,
                    'duration': 0
                })
                
                # Take initial snapshot
                self._take_violation_snapshot(frame, bbox, location_id, current_time)
                self.last_snapshot_time[location_id] = current_time
        
        # Clean up stale violations
        self._clean_stale_violations(current_time)

    def _clean_stale_violations(self, current_time):
        """Remove violations that haven't been detected for a while"""
        stale_threshold = 5  # Seconds to consider a violation stale
        
        alerts_to_remove = []
        for alert in self.illegal_alerts:
            time_since_last_seen = (current_time - alert['last_seen']).total_seconds()
            if time_since_last_seen > stale_threshold:
                alerts_to_remove.append(alert)
        
        for alert in alerts_to_remove:
       
            self.violations_tree.delete(alert['tree_id'])
            self.illegal_alerts.remove(alert)
            if alert['id'] in self.last_snapshot_time:
                del self.last_snapshot_time[alert['id']]
        
        if alerts_to_remove:
            self.update_status(f"Removed {len(alerts_to_remove)} stale violations")
        
    def _take_violation_snapshot(self, frame, bbox, location_id, timestamp):
        """Save a snapshot of the violation"""
        try:
            # Create violation snapshots directory if it doesn't exist
            snapshots_dir = os.path.join("output", "violations")
            os.makedirs(snapshots_dir, exist_ok=True)
            
            # Create snapshot with some margin around the vehicle
            x1, y1, x2, y2 = map(int, bbox)
            margin = 50
            h, w = frame.shape[:2]
            
            # Apply margin with bounds checking
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            snapshot = frame[y1:y2, x1:x2]
            
            # Save snapshot
            filename = f"violation_{location_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(snapshots_dir, filename)
            cv2.imwrite(filepath, snapshot)
            
        except Exception as e:
            self.update_status(f"Error saving violation snapshot: {str(e)}")
    
    def _update_parking_ui(self, frame, fps):
        """Update UI with latest frame and stats"""
        # Add FPS counter and other info to frame
        h, w = frame.shape[:2]
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        cv2.putText(
            frame, f"Violations: {len(self.illegal_alerts)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        
        # Display frame on UI
        display_image_on_label(frame, self.parking_video_label)
        
        # Update status bar
        self.update_status(f"Detecting parking violations | FPS: {fps:.1f} | Active violations: {len(self.illegal_alerts)}")
    
    def _stop_parking_detection(self):
        """Stop the parking detection process"""
        self.detection_running = False
        self.update_status("Stopping parking detection...")
    
    def _reset_parking_detection_ui(self):
        """Reset UI after detection stops"""
        self.parking_detect_btn.config(state=tk.NORMAL)
        self.parking_stop_btn.config(state=tk.DISABLED)
        self.update_status("Parking detection stopped")
    
    def _save_parking_violations(self):
        """Save detected parking violations to file"""
        if not self.illegal_alerts:
            messagebox.showinfo("Info", "No violations to save")
            return
            
        try:
            # Create output directory
            output_dir = os.path.join("output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parking_violations_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Format violation data
            violation_data = []
            for alert in self.illegal_alerts:
                violation_data.append({
                    "location": alert['center'],
                    "first_detected": alert['first_seen'].isoformat(),
                    "last_detected": alert['last_seen'].isoformat(),
                    "duration_seconds": alert['duration']
                })
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "video_source": self.parking_video_path_var.get(),
                    "zone_file": self.current_zone_file.get(),
                    "violations": violation_data
                }, f, indent=4)
            
            self.update_status(f"Saved {len(violation_data)} violations to {filename}")
            messagebox.showinfo("Success", f"Saved {len(violation_data)} violations to {filename}")
            
        except Exception as e:
            self.update_status(f"Error saving violations: {str(e)}")
            messagebox.showerror("Error", f"Failed to save violations: {str(e)}")

    def update_status(self, message, clear_after=None):
        """Update status message in the parking tab"""
        # First check if we have the parking_status_var attribute
        if hasattr(self, 'parking_status_var'):
            self.parking_status_var.set(message)
            self.root.update_idletasks()
            
            # If clear_after is specified, schedule clearing the status
            if clear_after is not None:
                self.root.after(clear_after, lambda: self.parking_status_var.set("Ready for parking detection"))
        else:
            # If we're still in initialization, just print the message
            print(f"Status: {message}")


if __name__ == "__main__":
    # Setup application root
    root = tk.Tk()
    root.title("Parking Violation Detection System")
    root.geometry("1280x800")
    
    # Initialize styles
    setup_styles(root)
    
    # Create app instance
    app = ParkingDetectionApp(root)
    
    # Run the application
    root.mainloop()