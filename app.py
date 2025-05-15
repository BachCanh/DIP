# app.py
import os
import cv2
import time
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, StringVar

# Import refactored components
import config
from car_detector import CarDetector
from ui_components import setup_styles, create_banner, setup_image_tab, setup_video_tab, create_status_bar, display_image_on_label
from video_processor import VideoProcessor

class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.detector = CarDetector() # Initialize detector

        # Check if model loaded successfully
        if not self.detector.model:
             messagebox.showerror("Fatal Error", "YOLO Model failed to load. Please check the path in config.py and Ultralytics installation.\nApplication will exit.")
             self.root.after(100, self.root.destroy) # Schedule closing
             return

        # --- State Variables ---
        self.image_path_var = StringVar()
        self.video_path_var = StringVar()
        self.output_dir_var = StringVar(value=config.DEFAULT_OUTPUT_DIR)
        self.status_var = StringVar(value="Ready.")
        self.confidence_threshold = config.DEFAULT_CONFIDENCE
        
        # Set all classes to be detected by default
        self.class_vars = {}
        for idx, class_name in enumerate(self.detector.class_names):
            self.class_vars[idx] = tk.BooleanVar(value=True)

        # Video processing state
        self.video_thread = None
        self.processing_video = False
        self.current_frame = None       # Last processed frame (image or video) for display/saving
        self.detected_objects = {}      # Final counts for saving (image or total video)

        # --- Setup UI ---
        self._setup_window()
        setup_styles(self.root)
        create_banner(self.root)
        self._create_main_interface()
        create_status_bar(self.root, self.status_var)
        self.center_window(root)

        self.update_status(self.detector.gpu_info, clear_after=5000) # Show GPU info briefly

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def center_window(self, root, width=None, height=None):
        root.update_idletasks()  # Make sure layout is updated

        if width is None or height is None:
            width = root.winfo_width()
            height = root.winfo_height()

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        root.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_window(self):
        self.root.title(config.PROJECT_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.configure(bg=config.BG_COLOR)
        # Add icon loading here if desired

    def _create_main_interface(self):
        main_frame = ttk.Frame(self.root, padding="10 10 10 10", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_frame, style="TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Create tabs using helper functions from ui_components
        # Pass `self` so the UI functions can access app's variables and methods
        image_tab = setup_image_tab(notebook, self)
        video_tab = setup_video_tab(notebook, self)

        notebook.add(image_tab, text=" Image Detection ")
        notebook.add(video_tab, text=" Video Detection ")

    def update_status(self, message, clear_after=0):
        self.status_var.set(message)
        if clear_after > 0:
            self.root.after(clear_after, lambda: self.status_var.set("Ready."))

    # --- File Browsing ---
    def browse_image(self):
        f_path = filedialog.askopenfilename(filetypes=(("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")))
        if f_path:
            self.image_path_var.set(f_path)
            self.update_status(f"Image selected: {os.path.basename(f_path)}")
            try:
                img = cv2.imread(f_path)
                if img is not None:
                    self.current_frame = img # Store original
                    display_image_on_label(img, self.image_display_label, max_width=self.image_display_label.winfo_width(), max_height=self.image_display_label.winfo_height())
                    self.img_result_text.delete(1.0, tk.END) # Clear old results
                else:
                    messagebox.showerror("Error", f"Failed to read image file:\n{f_path}")
            except Exception as e:
                 messagebox.showerror("Error", f"Error loading image preview:\n{e}")

    def browse_video(self):
        f_path = filedialog.askopenfilename(filetypes=(("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")))
        if f_path:
            self.video_path_var.set(f_path)
            self.update_status(f"Video selected: {os.path.basename(f_path)}")
            # Clear previous display/stats
            self.video_display_label.configure(image=None)
            self.video_display_label.image = None
            self.video_stats_text.delete(1.0, tk.END)
            self.video_progress["value"] = 0
            self.current_frame = None
            self.detected_objects = {}

    def browse_output_dir(self):
        diry = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if diry:
            self.output_dir_var.set(diry)
            self.update_status(f"Output directory set: {diry}")

    # --- Detection Logic ---
    def _get_selected_classes(self):
        return [idx for idx, var in self.class_vars.items() if var.get()]

    def detect_image(self):
        img_path = self.image_path_var.get()
        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("Error", "Please select a valid image file.")
            return

        selected_classes = self._get_selected_classes()
        if not selected_classes:
            messagebox.showwarning("No Classes", "No object classes selected for detection.")
            return

        self.update_status("Processing image...")
        self.root.update_idletasks()

        annotated_img, counts, error_msg = self.detector.predict_image(
            img_path, self.confidence_threshold, selected_classes
        )

        if error_msg:
            self.update_status(f"Error: {error_msg}")
            messagebox.showerror("Detection Error", error_msg)
            # Display original image if detection failed but image was readable
            original_img = cv2.imread(img_path)
            display_image_on_label(original_img, self.image_display_label, max_width=800, max_height=600)
            self.img_result_text.delete(1.0, tk.END)
            self.img_result_text.insert(tk.END, f"Detection failed: {error_msg}")
            self.current_frame = original_img # Store original on failure
            self.detected_objects = {}
            return

        # Success
        self.current_frame = annotated_img # Store annotated image
        self.detected_objects = {k: v for k, v in counts.items() if v > 0} # Store non-zero counts

        display_image_on_label(annotated_img, self.image_display_label, max_width=800, max_height=600)

        result_str = "--- Detection Summary ---\n\n"
        total = 0
        if not self.detected_objects:
             result_str += "(No objects detected)"
        else:
            for name, count in self.detected_objects.items():
                result_str += f"{name}: {count}\n"
                total += count
            result_str += f"\n--- Total Objects: {total} ---"

        self.img_result_text.delete(1.0, tk.END)
        self.img_result_text.insert(tk.END, result_str)
        self.update_status(f"Image processed. Found {total} objects.")

    # --- Video Processing ---
    def start_video_detection(self):
        if self.processing_video:
            messagebox.showwarning("Busy", "Video processing is already running.")
            return

        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        selected_classes = self._get_selected_classes()
        if not selected_classes:
            messagebox.showwarning("No Classes", "No object classes selected for detection.")
            return

        self.processing_video = True
        self._set_video_controls_state(processing=True)
        self.video_progress["value"] = 0
        self.video_stats_text.delete(1.0, tk.END)
        self.video_stats_text.insert(tk.END, "Starting detection...")
        self.detected_objects = {} # Reset results for new run

        self.update_status(f"Processing video: {os.path.basename(video_path)}")

        # Start the video processing thread
        self.video_thread = VideoProcessor(
            video_path=video_path,
            detector=self.detector,
            confidence=self.confidence_threshold,
            selected_classes=selected_classes,
            gui_update_callback=self._update_video_gui_callback,
            completion_callback=self._video_completion_callback,
            stop_callback=self._video_stop_callback
        )
        self.video_thread.start()

    def stop_video_detection(self):
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.stop() # Signal the thread to stop
            self.update_status("Stopping video processing...")
            self.stop_video_btn.configure(state=tk.DISABLED) # Prevent multiple clicks

    def _set_video_controls_state(self, processing: bool):
        """Enable/disable buttons based on processing state."""
        if processing:
            self.detect_video_btn.configure(state=tk.DISABLED)
            self.stop_video_btn.configure(state=tk.NORMAL)
            self.save_video_btn.configure(state=tk.DISABLED)
        else:
            self.detect_video_btn.configure(state=tk.NORMAL)
            self.stop_video_btn.configure(state=tk.DISABLED)
            # Enable save only if there's a frame/result
            self.save_video_btn.configure(state=tk.NORMAL if self.current_frame is not None else tk.DISABLED)


    def _update_video_gui_callback(self, frame, frame_counts, frame_num, total_frames, fps):
        """Callback executed by root.after from VideoProcessor thread."""
        if not self.processing_video: return # Avoid updates if stopped unexpectedly

        self.current_frame = frame # Store the latest annotated frame

        # Add FPS to the frame itself (optional)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        display_image_on_label(frame, self.video_display_label, max_width=800, max_height=600)

        # Update stats text (current frame)
        stats_str = "--- Current Frame ---\n"
        total_this_frame = 0
        filtered_counts = {k: v for k, v in frame_counts.items() if v > 0}
        if not filtered_counts:
            stats_str += "(No objects detected)\n"
        else:
            for name, count in filtered_counts.items():
                stats_str += f"{name}: {count}\n"
                total_this_frame += count
            stats_str += f"---------------------\nTotal: {total_this_frame}\n"

        self.video_stats_text.delete(1.0, tk.END)
        self.video_stats_text.insert(tk.END, stats_str)

        # Update progress bar
        if total_frames > 0:
            self.video_progress["value"] = frame_num
            self.video_progress["maximum"] = total_frames

    def _video_completion_callback(self, results):
        """Called when video finishes naturally."""
        if not self.processing_video: return # Already handled by stop?
        self.processing_video = False
        self.detected_objects = results # Store final accumulated counts
        self._set_video_controls_state(processing=False)
        self.video_progress["value"] = self.video_progress["maximum"]
        total_final = sum(results.values())
        self.update_status("Video processing complete.")
        messagebox.showinfo("Video Finished", f"Video processing finished.\nTotal objects (accumulated): {total_final}")

    def _video_stop_callback(self, results=None, error=None):
        """Called when video is stopped manually or errors."""
        self.processing_video = False
        if results:
             self.detected_objects = results # Store accumulated counts up to stop point
        self._set_video_controls_state(processing=False)
        if error:
             self.update_status(f"Video Error: {error}")
             messagebox.showerror("Video Error", error)
        else:
             total_final = sum(results.values()) if results else 0
             self.update_status("Video processing stopped by user.")
             messagebox.showinfo("Video Stopped", f"Video processing stopped.\nObjects detected (accumulated): {total_final}")

    # --- Saving Results ---
    def save_results(self, source_type):
        if self.current_frame is None:
            messagebox.showwarning("Save Result", "No result frame available to save.")
            return

        output_dir = self.output_dir_var.get()
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{source_type}_result_{timestamp}"
        img_filename = os.path.join(output_dir, f"{base_filename}.jpg")
        data_filename = os.path.join(output_dir, f"{base_filename}_data.json")

        try:
            # Save the image (current_frame is a numpy array)
            if not cv2.imwrite(img_filename, self.current_frame):
                 raise IOError(f"Failed to write image file: {img_filename}")

            # Prepare data
            data = {
                "timestamp": datetime.now().isoformat(),
                "source_type": source_type,
                "input_file": self.image_path_var.get() if source_type == "image" else self.video_path_var.get(),
                "settings": {
                    "confidence": self.confidence_threshold,
                    "detected_classes": [self.detector.class_names[i] for i in self._get_selected_classes()],
                },
                "detection_counts": self.detected_objects, # Contains image counts or total video counts
                "total_objects": sum(self.detected_objects.values()),
                "info": "Counts for video are accumulated over the processed duration." if source_type == "video" else "Counts for single image."
            }

            # Save data
            with open(data_filename, 'w') as f:
                json.dump(data, f, indent=4)

            self.update_status(f"Results saved to {output_dir}")
            messagebox.showinfo("Save Complete", f"Saved:\nImage: {os.path.basename(img_filename)}\nData: {os.path.basename(data_filename)}")

        except Exception as e:
             messagebox.showerror("Save Error", f"Failed to save results:\n{e}")
             self.update_status("Error saving results.")

    # --- Application Closing ---
    def on_closing(self):
        if self.processing_video:
            if messagebox.askyesno("Confirm Exit", "Video processing is running. Stop and exit?"):
                self.stop_video_detection()
                # Give the thread a moment to potentially stop
                if self.video_thread and self.video_thread.is_alive():
                    self.video_thread.join(timeout=0.5)
                self.root.destroy()
            else:
                return # Don't close
        else:
            self.root.destroy()