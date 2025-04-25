import os
import cv2
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, StringVar
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import csv
from datetime import datetime
import json

class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced YOLOv8 Object Detection Suite")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Load YOLOv8 model
        MODEL_PATH = "runs/train_cars_20250422_090512/yolov8n/weights/best.pt"
        self.model = YOLO(MODEL_PATH)
        
        # Get class names from model
        self.class_names = self.model.names
        
        # Initialize variables
        self.cap = None
        self.video_thread = None
        self.running = True
        self.processing_video = False
        self.detected_objects = {}
        self.detection_counts = {class_name: 0 for class_name in self.class_names.values()}
        self.confidence_threshold = 0.25
        self.current_frame = None
        self.detection_history = []
        self.recording_data = False
        self.data_recording_start_time = None
        
        # Create main frame with custom styling
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a notebook for tabs with custom styling
        style = ttk.Style()
        style.configure("TNotebook", background="#f0f0f0", borderwidth=0)
        style.configure("TNotebook.Tab", background="#d0d0d0", padding=[10, 5], font=('Arial', 10))
        style.map("TNotebook.Tab", background=[("selected", "#4a86e8")], foreground=[("selected", "white")])
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.image_tab = ttk.Frame(self.notebook)
        self.video_tab = ttk.Frame(self.notebook)
        self.camera_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.image_tab, text="Image Detection")
        self.notebook.add(self.video_tab, text="Video Detection")
        self.notebook.add(self.camera_tab, text="Camera Detection")
        self.notebook.add(self.analytics_tab, text="Analytics")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Setup tabs
        self.setup_image_tab()
        self.setup_video_tab()
        self.setup_camera_tab()
        self.setup_analytics_tab()
        self.setup_settings_tab()
        
        # Setup status bar
        self.status_var = StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load settings if available
        self.load_settings()
    
    def setup_image_tab(self):
        # Image file selection frame
        file_frame = ttk.LabelFrame(self.image_tab, text="Image Source")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Select Image File:").pack(side=tk.LEFT, padx=5)
        
        self.image_path_var = tk.StringVar()
        path_entry = ttk.Entry(file_frame, textvariable=self.image_path_var, width=50)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_image)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        detect_btn = ttk.Button(file_frame, text="Detect", command=self.detect_image)
        detect_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = ttk.Button(file_frame, text="Save Results", command=lambda: self.save_results("image"))
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display area with information panel
        display_frame = ttk.Frame(self.image_tab)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a PanedWindow for resizable sections
        paned = ttk.PanedWindow(display_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image display
        left_frame = ttk.Frame(paned)
        self.image_display = ttk.Label(left_frame)
        self.image_display.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Detection results
        right_frame = ttk.LabelFrame(paned, text="Detection Results")
        self.img_result_text = tk.Text(right_frame, width=40, height=20, wrap=tk.WORD)
        self.img_result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add frames to paned window
        paned.add(left_frame, weight=3)
        paned.add(right_frame, weight=1)
    
    def setup_video_tab(self):
        # Video file selection frame
        file_frame = ttk.LabelFrame(self.video_tab, text="Video Source")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Select Video File:").pack(side=tk.LEFT, padx=5)
        
        self.video_path_var = tk.StringVar()
        path_entry = ttk.Entry(file_frame, textvariable=self.video_path_var, width=50)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_video)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Control frame
        ctrl_frame = ttk.Frame(self.video_tab)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.detect_video_btn = ttk.Button(ctrl_frame, text="Start Detection", command=self.start_video_detection)
        self.detect_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_video_btn = ttk.Button(ctrl_frame, text="Stop Detection", command=self.stop_video_detection, state=tk.DISABLED)
        self.stop_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_video_btn = ttk.Button(ctrl_frame, text="Save Results", command=lambda: self.save_results("video"))
        self.save_video_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.video_progress = ttk.Progressbar(ctrl_frame, length=300, mode='determinate')
        self.video_progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Display area with information panel
        display_frame = ttk.Frame(self.video_tab)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a PanedWindow for resizable sections
        paned = ttk.PanedWindow(display_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video display
        left_frame = ttk.Frame(paned)
        self.video_display = ttk.Label(left_frame)
        self.video_display.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Live stats and controls
        right_frame = ttk.LabelFrame(paned, text="Live Detection Stats")
        
        # Object count display
        self.video_stats_text = tk.Text(right_frame, width=40, height=15, wrap=tk.WORD)
        self.video_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Record data checkbox
        self.record_data_var = tk.BooleanVar(value=False)
        record_check = ttk.Checkbutton(right_frame, text="Record Detection Data", 
                                     variable=self.record_data_var, 
                                     command=self.toggle_data_recording)
        record_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # Add frames to paned window
        paned.add(left_frame, weight=3)
        paned.add(right_frame, weight=1)
    
    def setup_camera_tab(self):
        # Camera control frame
        ctrl_frame = ttk.LabelFrame(self.camera_tab, text="Camera Controls")
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera selection
        ttk.Label(ctrl_frame, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.camera_id_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(ctrl_frame, textvariable=self.camera_id_var, width=5, 
                                   values=["0", "1", "2", "3"])
        camera_combo.pack(side=tk.LEFT, padx=5)
        
        self.start_cam_btn = ttk.Button(ctrl_frame, text="Start Camera", command=self.start_camera)
        self.start_cam_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_cam_btn = ttk.Button(ctrl_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_cam_btn.pack(side=tk.LEFT, padx=5)
        
        # Snapshot and recording controls
        self.snapshot_btn = ttk.Button(ctrl_frame, text="Take Snapshot", command=self.take_snapshot, state=tk.DISABLED)
        self.snapshot_btn.pack(side=tk.LEFT, padx=5)
        
        self.record_btn = ttk.Button(ctrl_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        # FPS display
        self.fps_var = StringVar(value="FPS: 0")
        fps_label = ttk.Label(ctrl_frame, textvariable=self.fps_var)
        fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Display area with information panel
        display_frame = ttk.Frame(self.camera_tab)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a PanedWindow for resizable sections
        paned = ttk.PanedWindow(display_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Camera display
        left_frame = ttk.Frame(paned)
        self.camera_display = ttk.Label(left_frame)
        self.camera_display.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Live stats and alerts
        right_frame = ttk.LabelFrame(paned, text="Live Detection Stats")
        
        # Object count display
        self.camera_stats_text = tk.Text(right_frame, width=40, height=15, wrap=tk.WORD)
        self.camera_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Alert settings
        alert_frame = ttk.LabelFrame(right_frame, text="Detection Alerts")
        alert_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Alert for specific class
        ttk.Label(alert_frame, text="Alert on:").pack(side=tk.LEFT, padx=5)
        self.alert_class_var = StringVar()
        alert_combo = ttk.Combobox(alert_frame, textvariable=self.alert_class_var, 
                                 values=list(self.class_names.values()), width=15)
        alert_combo.pack(side=tk.LEFT, padx=5)
        
        self.alert_enabled_var = tk.BooleanVar(value=False)
        alert_check = ttk.Checkbutton(alert_frame, text="Enable Alert", 
                                    variable=self.alert_enabled_var)
        alert_check.pack(side=tk.LEFT, padx=5)
        
        # Record data checkbox
        self.cam_record_data_var = tk.BooleanVar(value=False)
        record_check = ttk.Checkbutton(right_frame, text="Record Detection Data", 
                                     variable=self.cam_record_data_var,
                                     command=self.toggle_data_recording)
        record_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # Add frames to paned window
        paned.add(left_frame, weight=3)
        paned.add(right_frame, weight=1)
        
        # Recording variables
        self.is_recording = False
        self.video_writer = None
        self.last_frame_time = 0
        self.frame_times = []
    
    def setup_analytics_tab(self):
        # Analytics tab with data visualization
        top_frame = ttk.Frame(self.analytics_tab)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top_frame, text="Detection Analytics", font=("Arial", 16)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top_frame, text="Refresh Data", command=self.update_analytics).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top_frame, text="Export Analytics", command=self.export_analytics).pack(side=tk.RIGHT, padx=5)
        
        # Create tabs for different analytics views
        analytics_notebook = ttk.Notebook(self.analytics_tab)
        analytics_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary tab
        summary_tab = ttk.Frame(analytics_notebook)
        
        # Object counts chart frame
        chart_frame = ttk.LabelFrame(summary_tab, text="Detection Count by Class")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a figure for the chart
        self.analytics_figure, self.analytics_ax = plt.subplots(figsize=(8, 4))
        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_figure, master=chart_frame)
        self.analytics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Timeline tab
        timeline_tab = ttk.Frame(analytics_notebook)
        
        # Timeline chart frame
        timeline_frame = ttk.LabelFrame(timeline_tab, text="Detection Timeline")
        timeline_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a figure for the timeline
        self.timeline_figure, self.timeline_ax = plt.subplots(figsize=(8, 4))
        self.timeline_canvas = FigureCanvasTkAgg(self.timeline_figure, master=timeline_frame)
        self.timeline_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add tabs to analytics notebook
        analytics_notebook.add(summary_tab, text="Summary")
        analytics_notebook.add(timeline_tab, text="Timeline")
        
        # Initialize with empty data
        self.update_analytics()
    
    def setup_settings_tab(self):
        # Settings tab
        settings_frame = ttk.LabelFrame(self.settings_tab, text="Detection Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10, anchor=tk.N)
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.conf_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, length=300,
                                  command=self.update_conf_display)
        self.conf_scale.set(self.confidence_threshold)
        self.conf_scale.grid(row=0, column=1, padx=5, pady=5)
        
        self.conf_label = ttk.Label(settings_frame, text=f"{self.confidence_threshold:.2f}")
        self.conf_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Model selection
        ttk.Label(settings_frame, text="Model Type:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.model_var = StringVar(value="YOLOv8n")
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_var, 
                                  values=["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
        model_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Class selection for filtering
        ttk.Label(settings_frame, text="Classes to Detect:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.NW)
        
        # Create a frame for checkboxes
        class_frame = ttk.Frame(settings_frame)
        class_frame.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create checkboxes for each class
        self.class_vars = {}
        row, col = 0, 0
        for idx, class_name in self.class_names.items():
            self.class_vars[idx] = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(class_frame, text=class_name, variable=self.class_vars[idx])
            cb.grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            col += 1
            if col > 2:  # 3 columns of checkboxes
                col = 0
                row += 1
        
        # Save/Apply settings
        ttk.Button(settings_frame, text="Apply Settings", command=self.apply_settings).grid(row=3, column=0, padx=5, pady=15)
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(row=3, column=1, padx=5, pady=15)
        ttk.Button(settings_frame, text="Reset to Defaults", command=self.reset_settings).grid(row=3, column=2, padx=5, pady=15)
        
        # Output directory settings
        output_frame = ttk.LabelFrame(self.settings_tab, text="Output Settings")
        output_frame.pack(fill=tk.X, padx=10, pady=10, anchor=tk.N)
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.output_dir_var = StringVar(value="./output")
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50)
        output_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # About section
        about_frame = ttk.LabelFrame(self.settings_tab, text="About")
        about_frame.pack(fill=tk.X, padx=10, pady=10, anchor=tk.N)
        
        about_text = """Advanced YOLOv8 Object Detection Suite
Version 1.0.0
Built with YOLOv8 and Tkinter

This application provides powerful tools for object detection in images,
videos, and live camera feeds with comprehensive analytics and customizable settings.
"""
        about_label = ttk.Label(about_frame, text=about_text, justify=tk.LEFT)
        about_label.pack(padx=10, pady=10)
    
    def update_conf_display(self, value):
        conf = float(value)
        self.conf_label.config(text=f"{conf:.2f}")
        self.confidence_threshold = conf
    
    def apply_settings(self):
        # Apply settings without saving
        self.confidence_threshold = self.conf_scale.get()
        self.status_var.set(f"Settings applied. Confidence threshold: {self.confidence_threshold:.2f}")
    
    def save_settings(self):
        # Save settings to a file
        settings = {
            "confidence_threshold": self.confidence_threshold,
            "model_type": self.model_var.get(),
            "output_directory": self.output_dir_var.get(),
            "classes": {idx: var.get() for idx, var in self.class_vars.items()}
        }
        
        try:
            os.makedirs("./config", exist_ok=True)
            with open("./config/settings.json", "w") as f:
                json.dump(settings, f, indent=4)
            self.status_var.set("Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def load_settings(self):
        # Load settings from file
        try:
            if os.path.exists("./config/settings.json"):
                with open("./config/settings.json", "r") as f:
                    settings = json.load(f)
                
                # Apply loaded settings
                self.confidence_threshold = settings.get("confidence_threshold", 0.25)
                self.conf_scale.set(self.confidence_threshold)
                self.update_conf_display(self.confidence_threshold)
                
                self.model_var.set(settings.get("model_type", "YOLOv8n"))
                self.output_dir_var.set(settings.get("output_directory", "./output"))
                
                # Apply class settings if they exist
                if "classes" in settings:
                    for idx, value in settings["classes"].items():
                        if int(idx) in self.class_vars:
                            self.class_vars[int(idx)].set(value)
                
                self.status_var.set("Settings loaded successfully")
        except Exception as e:
            self.status_var.set(f"Failed to load settings: {str(e)}")
    
    def reset_settings(self):
        # Reset to default settings
        self.confidence_threshold = 0.25
        self.conf_scale.set(self.confidence_threshold)
        self.update_conf_display(self.confidence_threshold)
        
        self.model_var.set("YOLOv8n")
        self.output_dir_var.set("./output")
        
        # Reset all class checkboxes to True
        for var in self.class_vars.values():
            var.set(True)
        
        self.status_var.set("Settings reset to defaults")
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*"))
        )
        if file_path:
            self.image_path_var.set(file_path)
            self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
    
    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*"))
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.status_var.set(f"Video selected: {os.path.basename(file_path)}")
    
    def detect_image(self):
        img_path = self.image_path_var.get()
        if os.path.exists(img_path):
            self.status_var.set("Processing image...")
            
            # Get selected classes
            selected_classes = [int(idx) for idx, var in self.class_vars.items() if var.get()]
            
            # Reset detection counts
            self.detection_counts = {class_name: 0 for class_name in self.class_names.values()}
            
            # Perform detection
            results = self.model.predict(
                source=img_path, 
                conf=self.confidence_threshold,
                imgsz=640,
                classes=selected_classes
            )
            annotated = results[0].plot()
            
            # Update detection counts
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                if cls_name in self.detection_counts:
                    self.detection_counts[cls_name] += 1
            
            # Save the detection results for later use
            self.current_frame = annotated
            self.detected_objects = self.detection_counts.copy()
            
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated)
            
            # Resize while keeping aspect ratio
            display_width = 800
            ratio = display_width / pil_img.width
            display_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((display_width, display_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            img_tk = ImageTk.PhotoImage(pil_img)
            
            # Update display
            self.image_display.configure(image=img_tk)
            self.image_display.image = img_tk  # Keep a reference
            
            # Update results text
            self.img_result_text.delete(1.0, tk.END)
            self.img_result_text.insert(tk.END, "Detection Results:\n\n")
            
            # Add detection summary
            for class_name, count in self.detection_counts.items():
                if count > 0:
                    self.img_result_text.insert(tk.END, f"{class_name}: {count}\n")
            
            # Add confidence scores
            self.img_result_text.insert(tk.END, "\nConfidence Scores:\n")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                conf = float(box.conf[0])
                self.img_result_text.insert(tk.END, f"{cls_name}: {conf:.2f}\n")
            
            self.status_var.set(f"Image processed. Found {sum(self.detection_counts.values())} objects.")
            
            # Add to detection history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.detection_history.append({
                "timestamp": timestamp,
                "source": "image",
                "filename": os.path.basename(img_path),
                "counts": self.detection_counts.copy()
            })
    
    def start_video_detection(self):
        video_path = self.video_path_var.get()
        if os.path.exists(video_path) and not self.processing_video:
            self.processing_video = True
            self.running = True
            self.detect_video_btn.configure(state=tk.DISABLED)
            self.stop_video_btn.configure(state=tk.NORMAL)
            
            # Reset detection counts
            self.detection_counts = {class_name: 0 for class_name in self.class_names.values()}
            self.video_stats_text.delete(1.0, tk.END)
            
            # Get total frames (for progress bar)
            cap = cv2.VideoCapture(video_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Reset progress bar
            self.video_progress["value"] = 0
            self.video_progress["maximum"] = self.total_frames
            
            self.status_var.set(f"Processing video: {os.path.basename(video_path)}")
            
            # Start video processing in a separate thread
            self.video_thread = threading.Thread(target=self.process_video, args=(video_path,), daemon=True)
            self.video_thread.start()
            
            # Start data recording if selected
            if self.record_data_var.get():
                self.start_data_recording()
    
    def stop_video_detection(self):
        self.running = False
        self.status_var.set("Video processing stopped")
        if self.recording_data:
            self.stop_data_recording()
    
    def process_video(self, path):
        cap = cv2.VideoCapture(path)
        frame_count = 0
        start_time = time.time()
        frame_times = []
        
        # Get selected classes
        selected_classes = [int(idx) for idx, var in self.class_vars.items() if var.get()]
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Update progress bar every 5 frames to improve performance
            if frame_count % 5 == 0:
                self.root.after(1, lambda count=frame_count: self.video_progress.configure(value=count))
            
            # Calculate FPS
            frame_time = time.time()
            if len(frame_times) > 30:  # Keep only last 30 frames for FPS calculation
                frame_times.pop(0)
            frame_times.append(frame_time)
            
            # Perform detection
            results = self.model.predict(
                source=frame, 
                conf=self.confidence_threshold,
                imgsz=640,
                classes=selected_classes
            )
            annotated = results[0].plot()
            
            # Update detection counts
            frame_detections = {class_name: 0 for class_name in self.class_names.values()}
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                if cls_name in self.detection_counts:
                    self.detection_counts[cls_name] += 1
                    frame_detections[cls_name] += 1
            
            # Record data if enabled
            if self.recording_data:
                timestamp = time.time() - self.data_recording_start_time
                self.record_detection_data(timestamp, frame_detections, "video")
            
            # Save current frame and detections
            self.current_frame = annotated
            
            # Process in main thread to avoid Tkinter threading issues
            self.root.after(1, self.update_video_display, annotated, frame_times)
            
        cap.release()
        
        # Add to detection history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.detection_history.append({
            "timestamp": timestamp,
            "source": "video",
            "filename": os.path.basename(path),
            "counts": self.detection_counts.copy(),
            "frames": frame_count,
            "duration": time.time() - start_time
        })
        
        self.root.after(0, self.on_video_complete)
    
    def update_video_display(self, frame, frame_times):
        # Calculate FPS if we have enough frames
        if len(frame_times) > 1:
            fps = len(frame_times) / (frame_times[-1] - frame_times[0])
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Resize while keeping aspect ratio
        display_width = 800
        ratio = display_width / pil_img.width
        display_height = int(pil_img.height * ratio)
        pil_img = pil_img.resize((display_width, display_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        img_tk = ImageTk.PhotoImage(pil_img)
        
        # Update display
        self.video_display.configure(image=img_tk)
        self.video_display.image = img_tk  # Keep a reference
        
        # Update stats text
        self.video_stats_text.delete(1.0, tk.END)
        self.video_stats_text.insert(tk.END, "Detection Counts:\n\n")
        
        # Sort by count
        sorted_counts = {k: v for k, v in sorted(
            self.detection_counts.items(), 
            key=lambda item: item[1], 
            reverse=True
        ) if v > 0}
        
        for class_name, count in sorted_counts.items():
            self.video_stats_text.insert(tk.END, f"{class_name}: {count}\n")
    
    def on_video_complete(self):
        self.detect_video_btn.configure(state=tk.NORMAL)
        self.stop_video_btn.configure(state=tk.DISABLED)
        self.processing_video = False
        self.status_var.set("Video processing complete")
        
        # Stop data recording if it's active
        if self.recording_data:
            self.stop_data_recording()
        
        # Update analytics
        self.update_analytics()
    
    def start_camera(self):
        try:
            cam_id = int(self.camera_id_var.get())
            if self.cap is None:
                self.cap = cv2.VideoCapture(cam_id)
                if self.cap.isOpened():
                    self.start_cam_btn.configure(state=tk.DISABLED)
                    self.stop_cam_btn.configure(state=tk.NORMAL)
                    self.snapshot_btn.configure(state=tk.NORMAL)
                    self.record_btn.configure(state=tk.NORMAL)
                    
                    # Reset detection counts
                    self.detection_counts = {class_name: 0 for class_name in self.class_names.values()}
                    self.camera_stats_text.delete(1.0, tk.END)
                    
                    # Reset FPS calculation
                    self.frame_times = []
                    self.last_frame_time = time.time()
                    
                    # Start data recording if selected
                    if self.cam_record_data_var.get():
                        self.start_data_recording()
                    
                    self.status_var.set(f"Camera {cam_id} started")
                    self.update_camera()
                else:
                    self.cap = None
                    messagebox.showerror("Error", f"Could not open camera {cam_id}")
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.start_cam_btn.configure(state=tk.NORMAL)
            self.stop_cam_btn.configure(state=tk.DISABLED)
            self.snapshot_btn.configure(state=tk.DISABLED)
            self.record_btn.configure(state=tk.DISABLED)
            
            # Stop recording if active
            if self.is_recording and self.video_writer:
                self.toggle_recording()
            
            # Stop data recording if active
            if self.recording_data:
                self.stop_data_recording()
            
            # Clear display
            self.camera_display.configure(image='')
            self.status_var.set("Camera stopped")
            
            # Add to detection history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.detection_history.append({
                "timestamp": timestamp,
                "source": "camera",
                "device": self.camera_id_var.get(),
                "counts": self.detection_counts.copy()
            })
    
    def update_camera(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Calculate FPS
                current_time = time.time()
                self.frame_times.append(current_time)
                
                # Keep only last 30 frames for FPS calculation
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                
                if len(self.frame_times) > 1:
                    fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                    self.fps_var.set(f"FPS: {fps:.2f}")
                
                # Get selected classes
                selected_classes = [int(idx) for idx, var in self.class_vars.items() if var.get()]
                
                # Perform detection
                results = self.model.predict(
                    source=frame, 
                    conf=self.confidence_threshold,
                    imgsz=640,
                    classes=selected_classes
                )
                annotated = results[0].plot()
                
                # Check for alerts
                if self.alert_enabled_var.get():
                    alert_class = self.alert_class_var.get()
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.class_names[cls_id]
                        if cls_name == alert_class:
                            # Flash border or alert
                            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 0, 255), 10)
                            # Show alert message
                            if int(current_time) % 2 == 0:  # Flash every other second
                                cv2.putText(annotated, f"ALERT: {alert_class} detected", 
                                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Update detection counts for the current frame
                frame_detections = {class_name: 0 for class_name in self.class_names.values()}
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id]
                    if cls_name in self.detection_counts:
                        self.detection_counts[cls_name] += 1
                        frame_detections[cls_name] += 1
                
                # Record data if enabled
                if self.recording_data:
                    timestamp = time.time() - self.data_recording_start_time
                    self.record_detection_data(timestamp, frame_detections, "camera")
                
                # Save current frame for potential snapshot
                self.current_frame = annotated
                
                # Record video if active
                if self.is_recording and self.video_writer:
                    self.video_writer.write(annotated)
                
                # Convert to RGB for PIL
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(annotated_rgb)
                
                # Resize while keeping aspect ratio
                display_width = 800
                ratio = display_width / pil_img.width
                display_height = int(pil_img.height * ratio)
                pil_img = pil_img.resize((display_width, display_height), Image.LANCZOS)
                
                # Convert to PhotoImage
                img_tk = ImageTk.PhotoImage(pil_img)
                
                # Update display
                self.camera_display.configure(image=img_tk)
                self.camera_display.image = img_tk  # Keep a reference
                
                # Update stats
                self.camera_stats_text.delete(1.0, tk.END)
                self.camera_stats_text.insert(tk.END, "Current Detection Counts:\n\n")
                
                # Sort by count and show only non-zero counts
                sorted_counts = {k: v for k, v in sorted(
                    frame_detections.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                ) if v > 0}
                
                for class_name, count in sorted_counts.items():
                    self.camera_stats_text.insert(tk.END, f"{class_name}: {count}\n")
                
                self.camera_stats_text.insert(tk.END, "\nTotal Detections:\n")
                
                # Show total counts (since camera start)
                total_sorted = {k: v for k, v in sorted(
                    self.detection_counts.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                ) if v > 0}
                
                for class_name, count in total_sorted.items():
                    self.camera_stats_text.insert(tk.END, f"{class_name}: {count}\n")
                
                # Schedule next update
                self.root.after(30, self.update_camera)  # ~33 FPS max
            else:
                self.stop_camera()
        else:
            self.stop_camera()
    
    def take_snapshot(self):
        # Save the current frame as an image
        if self.current_frame is not None:
            # Ensure output directory exists
            output_dir = self.output_dir_var.get()
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/snapshot_{timestamp}.jpg"
            
            # Save image
            cv2.imwrite(filename, self.current_frame)
            self.status_var.set(f"Snapshot saved: {filename}")
            
            # Add to detection history
            self.detection_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "snapshot",
                "filename": os.path.basename(filename),
                "counts": self.detection_counts.copy()
            })
    
    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            output_dir = self.output_dir_var.get()
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/recording_{timestamp}.mp4"
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30.0  # Target fps
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            self.is_recording = True
            self.record_btn.configure(text="Stop Recording")
            self.status_var.set(f"Recording started: {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.is_recording = False
            self.record_btn.configure(text="Start Recording")
            self.status_var.set("Recording stopped")
    
    def toggle_data_recording(self):
        if not self.recording_data and (self.record_data_var.get() or self.cam_record_data_var.get()):
            self.start_data_recording()
        elif self.recording_data and not (self.record_data_var.get() or self.cam_record_data_var.get()):
            self.stop_data_recording()
    
    def start_data_recording(self):
        # Create directory for data if it doesn't exist
        output_dir = self.output_dir_var.get()
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_file = f"{output_dir}/data/detection_data_{timestamp}.csv"
        
        # Create CSV file with headers
        with open(self.data_file, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = ['timestamp', 'source'] + list(self.class_names.values()) + ['total']
            writer.writerow(headers)
        
        self.recording_data = True
        self.data_recording_start_time = time.time()
        self.status_var.set(f"Data recording started: {self.data_file}")
    
    def stop_data_recording(self):
        self.recording_data = False
        self.status_var.set("Data recording stopped")
        
        # Uncheck recording checkboxes
        self.record_data_var.set(False)
        self.cam_record_data_var.set(False)
    
    def record_detection_data(self, timestamp, detections, source):
        if not self.recording_data:
            return
            
        try:
            with open(self.data_file, 'a', newline='') as file:
                writer = csv.writer(file)
                
                # Prepare row data
                row_data = [timestamp, source]
                
                # Add counts for each class
                total_count = 0
                for class_name in self.class_names.values():
                    count = detections.get(class_name, 0)
                    row_data.append(count)
                    total_count += count
                
                # Add total count
                row_data.append(total_count)
                
                # Write to CSV
                writer.writerow(row_data)
        except Exception as e:
            print(f"Error recording data: {str(e)}")
            self.stop_data_recording()
    
    def update_analytics(self):
        # Update object count chart
        self.analytics_ax.clear()
        
        # Combine all detection counts from history
        if not self.detection_history:
            self.analytics_ax.text(0.5, 0.5, 'No detection data available', 
                                ha='center', va='center', transform=self.analytics_ax.transAxes)
        else:
            # Aggregate counts from all detection events
            combined_counts = {}
            for entry in self.detection_history:
                for cls, count in entry['counts'].items():
                    if cls not in combined_counts:
                        combined_counts[cls] = 0
                    combined_counts[cls] += count
            
            # Filter out zero counts and sort
            filtered_counts = {k: v for k, v in combined_counts.items() if v > 0}
            sorted_counts = dict(sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True))
            
            if sorted_counts:
                # Create bar chart
                classes = list(sorted_counts.keys())
                counts = list(sorted_counts.values())
                
                bars = self.analytics_ax.bar(classes, counts, color='skyblue')
                self.analytics_ax.set_ylabel('Count')
                self.analytics_ax.set_title('Detection Count by Class')
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    self.analytics_ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{int(height)}', ha='center', va='bottom')
                
                # Rotate x-axis labels for better readability
                self.analytics_ax.set_xticklabels(classes, rotation=45, ha='right')
                
                # Adjust layout
                self.analytics_figure.tight_layout()
            else:
                self.analytics_ax.text(0.5, 0.5, 'No non-zero detection counts available', 
                                    ha='center', va='center', transform=self.analytics_ax.transAxes)
        
        self.analytics_canvas.draw()
        
        # Update timeline chart
        self.timeline_ax.clear()
        
        if len(self.detection_history) > 1:
            # Create timeline data
            timestamps = []
            total_counts = []
            sources = []
            
            for entry in self.detection_history:
                # Convert timestamp string to datetime
                dt = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S")
                timestamps.append(dt)
                
                # Calculate total objects
                total = sum(entry['counts'].values())
                total_counts.append(total)
                
                # Record source
                sources.append(entry['source'])
            
            # Create scatter plot with different markers for different sources
            source_markers = {'image': 'o', 'video': 's', 'camera': '^', 'snapshot': 'x'}
            source_colors = {'image': 'blue', 'video': 'green', 'camera': 'red', 'snapshot': 'purple'}
            
            for source in set(sources):
                indices = [i for i, s in enumerate(sources) if s == source]
                self.timeline_ax.scatter(
                    [timestamps[i] for i in indices],
                    [total_counts[i] for i in indices],
                    marker=source_markers.get(source, 'o'),
                    color=source_colors.get(source, 'blue'),
                    label=source,
                    alpha=0.7
                )
            
            # Add line connecting points chronologically
            self.timeline_ax.plot(timestamps, total_counts, 'k-', alpha=0.3)
            
            self.timeline_ax.set_ylabel('Total Objects Detected')
            self.timeline_ax.set_title('Detection Timeline')
            
            # Format x-axis dates
            self.timeline_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
            
            # Rotate x-axis labels
            plt.setp(self.timeline_ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add legend
            self.timeline_ax.legend()
            
            # Adjust layout
            self.timeline_figure.tight_layout()
        else:
            self.timeline_ax.text(0.5, 0.5, 'Not enough data for timeline', 
                                ha='center', va='center', transform=self.timeline_ax.transAxes)
        
        self.timeline_canvas.draw()
    
    def export_analytics(self):
        # Export analytics data to CSV
        if not self.detection_history:
            messagebox.showinfo("Export Analytics", "No detection data available to export")
            return
        
        try:
            # Ensure output directory exists
            output_dir = self.output_dir_var.get()
            os.makedirs(f"{output_dir}/reports", exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/reports/analytics_report_{timestamp}.csv"
            
            # Create CSV file with detection history
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write headers
                headers = ['timestamp', 'source', 'filename']
                
                # Get all possible classes from history
                all_classes = set()
                for entry in self.detection_history:
                    all_classes.update(entry['counts'].keys())
                
                headers.extend(sorted(all_classes))
                headers.append('total')
                writer.writerow(headers)
                
                # Write data rows
                for entry in self.detection_history:
                    row = [
                        entry['timestamp'],
                        entry['source'],
                        entry.get('filename', entry.get('device', 'N/A'))
                    ]
                    
                    # Add counts for each class
                    for cls in sorted(all_classes):
                        row.append(entry['counts'].get(cls, 0))
                    
                    # Add total
                    row.append(sum(entry['counts'].values()))
                    
                    writer.writerow(row)
            
            # Generate summary report
            summary_filename = f"{output_dir}/reports/summary_report_{timestamp}.txt"
            with open(summary_filename, 'w') as file:
                file.write("YOLO Detection Summary Report\n")
                file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                file.write("Detection Sessions Summary:\n")
                file.write(f"Total sessions: {len(self.detection_history)}\n")
                
                # Count by source
                source_counts = {}
                for entry in self.detection_history:
                    source = entry['source']
                    if source not in source_counts:
                        source_counts[source] = 0
                    source_counts[source] += 1
                
                file.write("\nSessions by Source:\n")
                for source, count in source_counts.items():
                    file.write(f"- {source}: {count}\n")
                
                # Aggregate object counts
                combined_counts = {}
                for entry in self.detection_history:
                    for cls, count in entry['counts'].items():
                        if cls not in combined_counts:
                            combined_counts[cls] = 0
                        combined_counts[cls] += count
                
                file.write("\nTotal Objects Detected:\n")
                total_objects = sum(combined_counts.values())
                file.write(f"Total: {total_objects}\n\n")
                
                file.write("Objects by Class:\n")
                sorted_counts = dict(sorted(combined_counts.items(), key=lambda item: item[1], reverse=True))
                for cls, count in sorted_counts.items():
                    if count > 0:
                        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
                        file.write(f"- {cls}: {count} ({percentage:.1f}%)\n")
            
            self.status_var.set(f"Analytics exported to {filename} and {summary_filename}")
            messagebox.showinfo("Export Complete", f"Analytics data exported to:\n{filename}\n\nSummary report exported to:\n{summary_filename}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export analytics: {str(e)}")
    
    def save_results(self, source_type):
        # Save current detection results
        if self.current_frame is None:
            messagebox.showinfo("Save Results", "No detection results available to save")
            return
        
        try:
            # Ensure output directory exists
            output_dir = self.output_dir_var.get()
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{output_dir}/{source_type}_result_{timestamp}.jpg"
            
            # Save image with detections
            cv2.imwrite(image_filename, self.current_frame)
            
            # Save detection data as JSON
            data_filename = f"{output_dir}/{source_type}_data_{timestamp}.json"
            detection_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source_type,
                "counts": self.detected_objects,
                "total": sum(self.detected_objects.values()),
                "confidence_threshold": self.confidence_threshold
            }
            
            with open(data_filename, 'w') as f:
                json.dump(detection_data, f, indent=4)
            
            self.status_var.set(f"Results saved to {image_filename} and {data_filename}")
            messagebox.showinfo("Save Complete", f"Detection results saved to:\n{image_filename}\n\nData saved to:\n{data_filename}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")
    
    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        
        # Stop any recordings
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        
        # Stop data recording
        if self.recording_data:
            self.stop_data_recording()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectionApp(root)
    root.mainloop() 