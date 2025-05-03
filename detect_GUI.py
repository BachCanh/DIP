import os
import cv2
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, StringVar, font as tkFont
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import json
from datetime import datetime

class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Final Project: Car Detection")
        self.root.geometry("1280x850") # Slightly larger for better spacing
        self.root.configure(bg="#f0f0f0")
        # Initialize variables first
        self.status_var = tk.StringVar()  # Add this line before using status_var

        # --- Style Configuration ---
        self.setup_styles()

        # Create banner frame for project info
        self.create_banner_frame()

       # --- Check GPU availability ---
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.status_var.set(f"GPU detected: {gpu_name} ({gpu_memory:.2f} GB)")
            print(f"Using GPU: {gpu_name}")
        else:
            self.status_var.set("No GPU detected. Using CPU.")
            print("No GPU available. Using CPU.")
            
        # --- Load YOLOv8 model ---
        # Ensure this path is correct relative to where you run the script
        MODEL_PATH = "runs/train_cars_20250422_090512/yolov8n/weights/best.pt"
        try:
            self.model = YOLO(MODEL_PATH)
            # Configure the model to use GPU if available
            self.model.to(self.device)
            self.class_names = self.model.names
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load YOLO model from {MODEL_PATH}\nError: {e}\nPlease ensure the path is correct and Ultralytics is installed.")
            self.root.destroy()
            return

        # --- Initialize variables ---
        self.cap = None
        self.video_thread = None
        self.running = True
        self.processing_video = False
        self.detected_objects = {} # Stores final counts for saving (image or total video)
        self.detection_counts = {class_name: 0 for class_name in self.class_names.values()} # Accumulates total video counts
        self.confidence_threshold = 0.25
        self.current_frame = None # Holds the last processed frame (image or video) for display/saving

        # Create main frame
        main_frame = ttk.Frame(root, padding="10 10 10 10", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style="TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Create tabs (frames styled with Main.TFrame)
        self.image_tab = ttk.Frame(self.notebook, padding="5 5 5 5", style="Main.TFrame")
        self.video_tab = ttk.Frame(self.notebook, padding="5 5 5 5", style="Main.TFrame")
        self.settings_tab = ttk.Frame(self.notebook, padding="5 5 5 5", style="Main.TFrame")

        self.notebook.add(self.image_tab, text=" Image Detection ") # Added spaces for padding
        self.notebook.add(self.video_tab, text=" Video Detection ")
        self.notebook.add(self.settings_tab, text=" Settings ")

        # Setup tabs
        self.setup_image_tab()
        self.setup_video_tab()
        self.setup_settings_tab()

        # Setup status bar
        self.status_var = StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5 2", style="Status.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load settings if available
        self.load_settings()

    def setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use('clam') # Experiment with 'clam', 'alt', 'default'

        # Define fonts
        self.default_font = tkFont.nametofont("TkDefaultFont")
        self.default_font.configure(size=10)
        self.label_font = tkFont.Font(family="Arial", size=10)
        self.button_font = tkFont.Font(family="Arial", size=10, weight="bold")
        self.title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        self.banner_font = tkFont.Font(family="Arial", size=10)
        self.status_font = tkFont.Font(family="Arial", size=9)

        # Configure styles
        bg_color = "#f0f0f0"
        frame_bg = "#e0e0e0"
        accent_color = "#4a86e8"
        text_color = "#333333"
        button_fg = "white"

        style.configure(".", font=self.default_font, background=bg_color, foreground=text_color) # Global default

        style.configure("TFrame", background=bg_color)
        style.configure("Main.TFrame", background=bg_color) # Specific frame style if needed

        style.configure("TLabel", padding="5 5", font=self.label_font, background=bg_color)
        style.configure("Status.TLabel", font=self.status_font, background="#d0d0d0", foreground="#333333")
        style.configure("Conf.TLabel", font=self.label_font, background=frame_bg) # For confidence label

        style.configure("TButton", padding="8 4", font=self.button_font, foreground=button_fg, background=accent_color)
        style.map("TButton",
                  background=[('active', '#6faa Rfr') , ('pressed', '#3b6ec2')], # Slightly darker on hover/press
                  foreground=[('active', 'white')])

        style.configure("TEntry", padding="5 5", font=self.label_font)

        style.configure("TLabelframe", padding="10 10", font=self.label_font, background=frame_bg, relief=tk.GROOVE, borderwidth=1)
        style.configure("TLabelframe.Label", font=self.label_font, background=frame_bg, foreground=text_color)

        style.configure("TNotebook", background=bg_color, borderwidth=1)
        style.configure("TNotebook.Tab", padding=[12, 6], font=self.button_font, background="#d0d0d0", foreground="#555555")
        style.map("TNotebook.Tab",
                  background=[("selected", accent_color)],
                  foreground=[("selected", button_fg)])

        style.configure("TScale", background=frame_bg)
        style.configure("TProgressbar", thickness=20, background=accent_color, troughcolor='#d0d0d0')
        style.configure("TCheckbutton", background=frame_bg, font=self.label_font) # Ensure checkbuttons match frame bg
        style.configure("TCombobox", font=self.label_font, padding="5 5")


    def create_banner_frame(self):
        banner_bg = "#1e3c72"
        banner_fg = "white"
        banner_frame = tk.Frame(self.root, bg=banner_bg) # Use tk.Frame for simple color bg
        banner_frame.pack(fill=tk.X, pady=(0, 10))

        # Project title (centered)
        title_label = tk.Label(banner_frame, text="Car Detection - Final Project",
                               font=self.title_font, fg=banner_fg, bg=banner_bg)
        title_label.pack(side=tk.LEFT, padx=30, pady=15, expand=True) # Expand to help center

        # Team members frame
        team_frame = tk.Frame(banner_frame, bg=banner_bg)
        team_frame.pack(side=tk.RIGHT, padx=20, pady=10)

        # Team members
        team_text = (
            "Team:\n"
            "Bạch Đức Cảnh (22110012) | Nguyễn Tiến Toàn (22110078)\n"
            "Lý Đăng Triều (22110080) | Nguyễn Tuấn Vũ (22110091)"
        )
        team_label = tk.Label(team_frame, text=team_text, font=self.banner_font,
                              fg=banner_fg, bg=banner_bg, justify=tk.RIGHT)
        team_label.pack()

    def setup_image_tab(self):
        # Configure grid columns for expansion
        self.image_tab.columnconfigure(0, weight=1)
        self.image_tab.rowconfigure(1, weight=1) # Allow display frame to expand

        # --- Image file selection frame ---
        file_frame = ttk.LabelFrame(self.image_tab, text="Image Source", padding="10 10")
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))
        file_frame.columnconfigure(1, weight=1) # Make entry expand

        ttk.Label(file_frame, text="Select Image:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")

        self.image_path_var = tk.StringVar()
        path_entry = ttk.Entry(file_frame, textvariable=self.image_path_var, width=60)
        path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_image)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)

        detect_btn = ttk.Button(file_frame, text="Detect Objects", command=self.detect_image)
        detect_btn.grid(row=0, column=3, padx=5, pady=5)

        save_btn = ttk.Button(file_frame, text="Save Result", command=lambda: self.save_results("image"))
        save_btn.grid(row=0, column=4, padx=5, pady=5)

        # --- Image display area with information panel ---
        display_frame = ttk.Frame(self.image_tab, style="Main.TFrame") # Use main background
        display_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0,5))

        # PanedWindow for resizable sections
        paned = ttk.PanedWindow(display_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Image display (add a frame for potential padding/border)
        left_frame_outer = ttk.Frame(paned, style="Main.TFrame")
        self.image_display = ttk.Label(left_frame_outer, background='black', anchor=tk.CENTER) # Black background for image area
        self.image_display.pack(fill=tk.BOTH, expand=True, padx=1, pady=1) # Small padding inside

        # Right panel - Detection results
        right_frame = ttk.LabelFrame(paned, text="Detection Results", padding="10 10")
        self.img_result_text = tk.Text(right_frame, width=35, height=20, wrap=tk.WORD, font=self.label_font, relief=tk.SOLID, borderwidth=1)
        self.img_result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add frames to paned window (adjust weight as needed)
        paned.add(left_frame_outer, weight=3)
        paned.add(right_frame, weight=1)

    def setup_video_tab(self):
        # Configure grid columns/rows
        self.video_tab.columnconfigure(0, weight=1)
        self.video_tab.rowconfigure(2, weight=1) # Allow display frame to expand

        # --- Video file selection frame ---
        file_frame = ttk.LabelFrame(self.video_tab, text="Video Source", padding="10 10")
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))
        file_frame.columnconfigure(1, weight=1) # Make entry expand

        ttk.Label(file_frame, text="Select Video:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")

        self.video_path_var = tk.StringVar()
        path_entry = ttk.Entry(file_frame, textvariable=self.video_path_var, width=60)
        path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_video)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)

        # --- Control frame ---
        ctrl_frame = ttk.Frame(self.video_tab, padding="5 0", style="Main.TFrame")
        ctrl_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 10))
        ctrl_frame.columnconfigure(3, weight=1) # Make progress bar expand

        self.detect_video_btn = ttk.Button(ctrl_frame, text="Start Detection", command=self.start_video_detection)
        self.detect_video_btn.grid(row=0, column=0, padx=(0, 5), pady=5)

        self.stop_video_btn = ttk.Button(ctrl_frame, text="Stop Detection", command=self.stop_video_detection, state=tk.DISABLED)
        self.stop_video_btn.grid(row=0, column=1, padx=5, pady=5)

        self.save_video_btn = ttk.Button(ctrl_frame, text="Save Last Frame", command=lambda: self.save_results("video"))
        self.save_video_btn.grid(row=0, column=2, padx=5, pady=5)

        # Progress bar
        self.video_progress = ttk.Progressbar(ctrl_frame, length=300, mode='determinate', style="TProgressbar")
        self.video_progress.grid(row=0, column=3, padx=10, pady=5, sticky="ew")

        # --- Display area ---
        display_frame = ttk.Frame(self.video_tab, style="Main.TFrame")
        display_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=(0,5))

        paned = ttk.PanedWindow(display_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Video display
        left_frame_outer = ttk.Frame(paned, style="Main.TFrame")
        self.video_display = ttk.Label(left_frame_outer, background='black', anchor=tk.CENTER)
        self.video_display.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Right panel - Live stats
        right_frame = ttk.LabelFrame(paned, text="Live Frame Stats", padding="10 10")
        self.video_stats_text = tk.Text(right_frame, width=35, height=15, wrap=tk.WORD, font=self.label_font, relief=tk.SOLID, borderwidth=1)
        self.video_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        paned.add(left_frame_outer, weight=3)
        paned.add(right_frame, weight=1)


    def setup_settings_tab(self):
        settings_scroll_frame = tk.Frame(self.settings_tab, bg=self.root.cget('bg')) # Match root bg
        settings_scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas and a vertical scrollbar for scrolling the settings
        canvas = tk.Canvas(settings_scroll_frame, bg=self.root.cget('bg'), highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Main.TFrame") # Use the styled frame

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Detection Settings ---
        settings_frame = ttk.LabelFrame(scrollable_frame, text="Detection Settings", padding="15 15")
        settings_frame.pack(fill=tk.X, padx=10, pady=10, anchor=tk.N)
        settings_frame.columnconfigure(1, weight=1) # Allow scale/combobox to expand slightly

        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        self.conf_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, length=300,
                                    orient=tk.HORIZONTAL, command=self.update_conf_display)
        self.conf_scale.set(self.confidence_threshold)
        self.conf_scale.grid(row=0, column=1, padx=5, pady=10, sticky=tk.EW)
        self.conf_label = ttk.Label(settings_frame, text=f"{self.confidence_threshold:.2f}", style="Conf.TLabel", width=5) # Fixed width
        self.conf_label.grid(row=0, column=2, padx=5, pady=10)

        # Model selection (Note: Reloading model based on this is not implemented)
        ttk.Label(settings_frame, text="Model Type:").grid(row=1, column=0, padx=5, pady=10, sticky=tk.W)
        self.model_var = StringVar(value="YOLOv8n (Loaded)") # Indicate loaded model
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_var, state="readonly", # Make readonly as switching isn't implemented
                                   values=["YOLOv8n (Loaded)", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
        model_combo.grid(row=1, column=1, columnspan=2, padx=5, pady=10, sticky=tk.W)
        # TODO: Add functionality to reload the model when selection changes if desired.

        # --- Class selection for filtering ---
        ttk.Label(settings_frame, text="Filter Classes:").grid(row=2, column=0, padx=5, pady=10, sticky=tk.NW)
        class_frame = ttk.Frame(settings_frame, style="TLabelframe") # Use frame background for checkboxes
        class_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=10, sticky=tk.W)

        self.class_vars = {}
        num_cols = 4 # Adjust columns for checkboxes
        row, col = 0, 0
        for idx, class_name in self.class_names.items():
            self.class_vars[idx] = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(class_frame, text=class_name.capitalize(), variable=self.class_vars[idx])
            cb.grid(row=row, column=col, padx=5, pady=3, sticky=tk.W)
            col += 1
            if col >= num_cols:
                col = 0
                row += 1

        # --- Buttons for Settings ---
        button_frame = ttk.Frame(settings_frame, style="TLabelframe") # Use frame background
        button_frame.grid(row=3, column=0, columnspan=3, pady=15)

        ttk.Button(button_frame, text="Apply Settings", command=self.apply_settings).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Reset Defaults", command=self.reset_settings).pack(side=tk.LEFT, padx=10)


        # --- Output directory settings ---
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Settings", padding="15 15")
        output_frame.pack(fill=tk.X, padx=10, pady=10, anchor=tk.N)
        output_frame.columnconfigure(1, weight=1) # Make entry expand

        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_dir_var = StringVar(value="./output")
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=60)
        output_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir).grid(row=0, column=2, padx=5, pady=5)


        # --- About section ---
        about_frame = ttk.LabelFrame(scrollable_frame, text="About", padding="15 15")
        about_frame.pack(fill=tk.X, padx=10, pady=10, anchor=tk.N)

        about_text = """YOLO Object Detection App - v1.1
---------------------------------------
Built with YOLOv8 and Tkinter.

Detects objects in images and videos using a trained YOLO model.
Features include confidence adjustment, class filtering, and results saving.

Ho Chi Minh City University of Technology and Education
Faculty of Electrical and Electronics Engineering
Department of Computer Engineering
Final Project - Computer Vision
"""
        about_label = ttk.Label(about_frame, text=about_text, justify=tk.LEFT, style="TLabelframe.Label") # Match frame style
        about_label.pack(padx=10, pady=10, anchor=tk.W)

    def update_conf_display(self, value):
        # Callback for confidence scale
        conf = float(value)
        self.conf_label.config(text=f"{conf:.2f}")
        # No need to set self.confidence_threshold here, apply_settings does that if needed.

    def apply_settings(self):
        # Apply settings immediately without saving to file
        self.confidence_threshold = self.conf_scale.get()
        # Add logic here if model switching was implemented
        self.status_var.set(f"Settings applied. Confidence: {self.confidence_threshold:.2f}. Output: {self.output_dir_var.get()}")
        messagebox.showinfo("Settings Applied", "Current detection settings have been updated.")


    def save_settings(self):
        # Apply current settings from UI first
        self.confidence_threshold = self.conf_scale.get()

        settings = {
            "confidence_threshold": self.confidence_threshold,
            "model_type": self.model_var.get(), # Save selected model name
            "output_directory": self.output_dir_var.get(),
            # Store class filter settings (using int keys for JSON compatibility if needed)
            "classes": {str(idx): var.get() for idx, var in self.class_vars.items()}
        }

        try:
            config_dir = "./config"
            os.makedirs(config_dir, exist_ok=True)
            settings_path = os.path.join(config_dir, "settings.json")
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=4)
            self.status_var.set(f"Settings saved successfully to {settings_path}")
            messagebox.showinfo("Settings Saved", f"Settings saved to {settings_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            self.status_var.set(f"Error saving settings: {str(e)}")

    def load_settings(self):
        settings_path = "./config/settings.json"
        try:
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    settings = json.load(f)

                # Apply loaded settings
                self.confidence_threshold = settings.get("confidence_threshold", 0.25)
                self.conf_scale.set(self.confidence_threshold)
                self.update_conf_display(self.confidence_threshold) # Update label too

                # Note: Model loading based on 'model_type' is not implemented here.
                # self.model_var.set(settings.get("model_type", "YOLOv8n (Loaded)"))

                self.output_dir_var.set(settings.get("output_directory", "./output"))

                # Apply class filter settings
                loaded_classes = settings.get("classes", {})
                for idx_str, value in loaded_classes.items():
                    try:
                        idx = int(idx_str)
                        if idx in self.class_vars:
                            self.class_vars[idx].set(bool(value))
                    except ValueError:
                         print(f"Warning: Invalid class index '{idx_str}' in settings file.")


                self.status_var.set("Settings loaded successfully")
            else:
                 self.status_var.set("Settings file not found. Using defaults.")
                 self.reset_settings(show_message=False) # Apply defaults if no file

        except json.JSONDecodeError as e:
            messagebox.showerror("Settings Load Error", f"Error decoding settings file: {settings_path}\n{e}")
            self.status_var.set(f"Error loading settings file: {e}")
            self.reset_settings(show_message=False) # Reset on bad file
        except Exception as e:
            messagebox.showerror("Settings Load Error", f"Failed to load settings: {str(e)}")
            self.status_var.set(f"Failed to load settings: {str(e)}")
            self.reset_settings(show_message=False) # Reset on other errors

    def reset_settings(self, show_message=True):
        # Reset UI elements to defaults
        default_conf = 0.25
        self.conf_scale.set(default_conf)
        self.update_conf_display(default_conf)
        self.confidence_threshold = default_conf # Also update the internal variable

        # self.model_var.set("YOLOv8n (Loaded)") # Reset model display
        self.output_dir_var.set("./output")

        # Reset all class checkboxes to True
        for var in self.class_vars.values():
            var.set(True)

        self.status_var.set("Settings reset to defaults")
        if show_message:
            messagebox.showinfo("Settings Reset", "All settings have been reset to their default values.")


    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory", initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
            self.status_var.set(f"Output directory set to: {directory}")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=(("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All Files", "*.*"))
        )
        if file_path:
            self.image_path_var.set(file_path)
            self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
            # Optionally display the selected image immediately (without detection)
            self.display_image(file_path)
            self.img_result_text.delete(1.0, tk.END) # Clear previous results

    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*"))
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.status_var.set(f"Video selected: {os.path.basename(file_path)}")
            # Clear previous video display and stats
            self.video_display.configure(image=None)
            self.video_display.image = None
            self.video_stats_text.delete(1.0, tk.END)
            self.video_progress["value"] = 0


    def display_image(self, img_path_or_array, is_detected=False):
        """ Helper to display an image (from path or numpy array) """
        try:
            if isinstance(img_path_or_array, str): # If it's a path, load it
                img = cv2.imread(img_path_or_array)
                if img is None:
                    raise ValueError(f"Could not read image file: {img_path_or_array}")
            else: # Assume it's a numpy array (already loaded/processed)
                img = img_path_or_array

            # Convert BGR (OpenCV) to RGB (PIL)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Resize for display, maintaining aspect ratio
            # Get the display widget's size for better fitting
            widget_width = self.image_display.winfo_width()
            widget_height = self.image_display.winfo_height()

            # Fallback size if widget size not available yet
            if widget_width <= 1: widget_width = 800
            if widget_height <= 1: widget_height = 600

            img_width, img_height = pil_img.size
            ratio = min(widget_width / img_width, widget_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            # Only resize if needed (don't upscale small images too much unless necessary)
            if new_width < img_width or new_height < img_height or not is_detected : # Resize detected or if original is larger than display
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)


            # Convert to PhotoImage
            img_tk = ImageTk.PhotoImage(pil_img)

            # Update display
            self.image_display.configure(image=img_tk)
            self.image_display.image = img_tk  # Keep a reference!
            if not is_detected:
                 self.current_frame = img # Store original cv2 image if just displaying
            else:
                 self.current_frame = img # Store detected cv2 image

        except FileNotFoundError:
             messagebox.showerror("Error", f"Image file not found: {img_path_or_array}")
             self.status_var.set("Error: Image file not found.")
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
            self.status_var.set(f"Error: {ve}")
        except Exception as e:
            messagebox.showerror("Image Display Error", f"Failed to display image.\nError: {e}")
            self.status_var.set(f"Error displaying image: {e}")

    def detect_image(self):
        img_path = self.image_path_var.get()
        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("Error", "Please select a valid image file first.")
            return

        self.status_var.set("Processing image...")
        self.root.update_idletasks() # Update GUI to show status

        # Get selected classes to detect based on checkboxes
        selected_classes_idx = [idx for idx, var in self.class_vars.items() if var.get()]
        if not selected_classes_idx:
            messagebox.showwarning("No Classes Selected", "Please select at least one class to detect in the Settings tab.")
            self.status_var.set("Ready")
            return

        # Reset detection counts for this specific image run
        image_detection_counts = {class_name: 0 for class_name in self.class_names.values()}
        confidence_scores = [] # Store confidences for this image

        try:
            # Perform detection
            results = self.model.predict(
                source=img_path,
                conf=self.confidence_threshold,
                imgsz=640,
                classes=selected_classes_idx # Filter by selected classes
            )

            # Check if results were found
            if not results or len(results) == 0 or results[0].boxes is None:
                 messagebox.showinfo("Detection", "No objects detected in the image with the current settings.")
                 self.status_var.set("Image processed. No objects found.")
                 self.display_image(img_path) # Show original image
                 self.img_result_text.delete(1.0, tk.END)
                 self.img_result_text.insert(tk.END, "No objects detected.")
                 self.detected_objects = {} # Clear previous results
                 return


            annotated_frame = results[0].plot() # Get the annotated image (numpy array)

            # Update detection counts and collect scores for this image
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"Unknown({cls_id})")
                conf = float(box.conf[0])

                if cls_name in image_detection_counts: # Should always be true if class_names is correct
                    image_detection_counts[cls_name] += 1
                confidence_scores.append((cls_name, conf))


            # --- Update GUI ---
            # Display the annotated image
            self.display_image(annotated_frame, is_detected=True)

            # Update results text
            self.img_result_text.delete(1.0, tk.END)
            self.img_result_text.insert(tk.END, "--- Detection Summary ---\n\n")
            total_detections = 0
            for class_name, count in image_detection_counts.items():
                if count > 0:
                    self.img_result_text.insert(tk.END, f"{class_name}: {count}\n")
                    total_detections += count

            self.img_result_text.insert(tk.END, f"\n--- Total Objects: {total_detections} ---\n")

            # Add confidence scores (optional, can make the box long)
            # self.img_result_text.insert(tk.END, "\n--- Confidence Scores ---\n")
            # confidence_scores.sort(key=lambda x: x[1], reverse=True) # Sort by confidence
            # for name, conf in confidence_scores:
            #     self.img_result_text.insert(tk.END, f"{name}: {conf:.3f}\n")

            # Store results for potential saving
            self.detected_objects = image_detection_counts.copy()
            # self.current_frame is already set by display_image

            self.status_var.set(f"Image processed. Found {total_detections} objects.")

        except Exception as e:
            messagebox.showerror("Detection Error", f"An error occurred during image detection:\n{e}")
            self.status_var.set(f"Error during detection: {e}")


    def start_video_detection(self):
        video_path = self.video_path_var.get()
        if self.processing_video:
             messagebox.showwarning("Busy", "Video processing is already in progress.")
             return
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file first.")
            return

        # Check if classes are selected
        self.selected_classes_idx_video = [idx for idx, var in self.class_vars.items() if var.get()] # Store for thread
        if not self.selected_classes_idx_video:
            messagebox.showwarning("No Classes Selected", "Please select at least one class to detect in the Settings tab.")
            return

        self.processing_video = True
        self.running = True # Ensure running flag is true
        self.detect_video_btn.configure(state=tk.DISABLED)
        self.stop_video_btn.configure(state=tk.NORMAL)
        self.save_video_btn.configure(state=tk.DISABLED) # Disable saving until done/stopped

        # Reset TOTAL detection counts for the new video run
        self.detection_counts = {class_name: 0 for class_name in self.class_names.values()}
        self.video_stats_text.delete(1.0, tk.END)
        self.video_stats_text.insert(tk.END, "Starting detection...")

        # Get total frames for progress bar
        try:
            temp_cap = cv2.VideoCapture(video_path)
            if not temp_cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")
            self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_cap.release()
        except Exception as e:
            messagebox.showerror("Video Error", f"Could not get video properties.\nError: {e}")
            self.processing_video = False
            self.detect_video_btn.configure(state=tk.NORMAL)
            self.stop_video_btn.configure(state=tk.DISABLED)
            return

        # Reset progress bar
        self.video_progress["value"] = 0
        self.video_progress["maximum"] = self.total_frames if self.total_frames > 0 else 1 # Avoid division by zero

        self.status_var.set(f"Processing video: {os.path.basename(video_path)}")

        # Start video processing in a separate thread
        # Pass necessary parameters that might change (like conf threshold, classes)
        self.video_thread = threading.Thread(target=self.process_video,
                                             args=(video_path, self.confidence_threshold, self.selected_classes_idx_video),
                                             daemon=True) # Daemon thread exits if main app closes
        self.video_thread.start()

    def stop_video_detection(self):
        if self.processing_video and self.running:
            self.running = False # Signal the thread to stop
            self.status_var.set("Stopping video processing...")
            self.stop_video_btn.configure(state=tk.DISABLED) # Prevent multiple clicks
            # The thread will finish its current loop and call on_video_complete/on_video_stop
            # Wait a short moment for the thread to potentially finish gracefully (optional)
            # if self.video_thread and self.video_thread.is_alive():
            #     self.video_thread.join(timeout=1.0) # Wait up to 1 second

    def process_video(self, path, current_conf_threshold, current_selected_classes):
        """ Video processing function running in a separate thread """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
             self.root.after(0, lambda: messagebox.showerror("Video Error", f"Cannot open video file in thread: {path}"))
             self.root.after(0, self.on_video_stop) # Use stop state handler
             return

        frame_count = 0
        start_time = time.time()
        # frame_times = [] # Removed FPS calculation from here, moved to main thread update

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.running = False # End of video
                break

            frame_count += 1
            loop_start_time = time.time()

            # --- Perform detection ---
            results = self.model.predict(
                source=frame,
                conf=current_conf_threshold, # Use threshold passed at start
                imgsz=640,
                classes=current_selected_classes, # Use classes passed at start
                verbose=False # Suppress console output from predict
            )

            # Check results format
            if not results or len(results) == 0:
                 annotated_frame = frame # No detections, use original frame
                 frame_detections = {name: 0 for name in self.class_names.values()} # Zero counts for this frame
            else:
                 annotated_frame = results[0].plot() # Get annotated frame

                 # --- Calculate counts for THIS frame ---
                 frame_detections = {name: 0 for name in self.class_names.values()}
                 for box in results[0].boxes:
                     cls_id = int(box.cls[0])
                     cls_name = self.class_names.get(cls_id, f"Unknown({cls_id})")
                     if cls_name in frame_detections:
                         frame_detections[cls_name] += 1
                         # --- Accumulate TOTAL counts ---
                         if cls_name in self.detection_counts:
                              self.detection_counts[cls_name] += 1


            # --- Schedule GUI update in main thread ---
            # Pass the annotated frame and the counts FOR THIS FRAME
            self.root.after(1, self.update_video_display, annotated_frame, frame_detections, frame_count, loop_start_time)

            # Small sleep to prevent overwhelming the GUI thread (optional, adjust as needed)
            # time.sleep(0.01)


        # --- Cleanup ---
        cap.release()

        # --- Final GUI update call ---
        # Check if stopped manually or finished naturally
        if not self.running and frame_count < self.total_frames: # Stopped manually
             self.root.after(0, self.on_video_stop)
        else: # Finished video
            self.root.after(0, self.on_video_complete)

    def update_video_display(self, annotated_frame, frame_detections, frame_number, frame_start_time):
        """ Updates the video display and stats text in the main GUI thread """
        if not self.processing_video and not self.running: # Check if we stopped abruptly
             return # Avoid updating if processing is meant to be stopped

        # --- FPS Calculation ---
        processing_time = time.time() - frame_start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # --- Store current frame for potential saving ---
        self.current_frame = annotated_frame # Store the annotated frame

        # --- Convert and resize for display ---
        try:
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            widget_width = self.video_display.winfo_width()
            widget_height = self.video_display.winfo_height()
            if widget_width <= 1: widget_width = 800
            if widget_height <= 1: widget_height = 600

            img_width, img_height = pil_img.size
            ratio = min(widget_width / img_width, widget_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            if new_width < img_width or new_height < img_height: # Only downscale
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(pil_img)

            # --- Update Video Display ---
            self.video_display.configure(image=img_tk)
            self.video_display.image = img_tk  # Keep reference

            # --- Update Stats Text (Counts for CURRENT frame) ---
            self.video_stats_text.delete(1.0, tk.END)
            self.video_stats_text.insert(tk.END, "--- Current Frame Detections ---\n\n")

            total_frame_detections = 0
            # Sort frame detections by count desc
            sorted_frame_counts = {k: v for k, v in sorted(frame_detections.items(), key=lambda item: item[1], reverse=True) if v > 0}

            if not sorted_frame_counts:
                 self.video_stats_text.insert(tk.END, "(No objects detected in this frame)")
            else:
                for class_name, count in sorted_frame_counts.items():
                    self.video_stats_text.insert(tk.END, f"{class_name}: {count}\n")
                    total_frame_detections += count
                self.video_stats_text.insert(tk.END, f"\n--- Total This Frame: {total_frame_detections} ---")


            # --- Update Progress Bar ---
            if self.total_frames > 0:
                 self.video_progress["value"] = frame_number

        except Exception as e:
             print(f"Error updating video display: {e}") # Print error, but try to continue
             # Avoid showing messagebox here as it could flood the user


    def on_video_complete(self):
        """ Called when video processing finishes naturally """
        self.processing_video = False
        self.running = False # Ensure flag is off
        self.detect_video_btn.configure(state=tk.NORMAL)
        self.stop_video_btn.configure(state=tk.DISABLED)
        self.save_video_btn.configure(state=tk.NORMAL) # Enable saving final results
        self.video_progress["value"] = self.video_progress["maximum"] # Fill progress bar
        self.status_var.set("Video processing complete.")
        # Store the TOTAL accumulated counts for saving
        self.detected_objects = self.detection_counts.copy()
        messagebox.showinfo("Video Finished", f"Video processing finished.\nTotal objects detected (accumulated): {sum(self.detected_objects.values())}")

    def on_video_stop(self):
        """ Called when video processing is stopped manually """
        self.processing_video = False
        self.running = False # Ensure flag is off
        self.detect_video_btn.configure(state=tk.NORMAL)
        self.stop_video_btn.configure(state=tk.DISABLED)
        self.save_video_btn.configure(state=tk.NORMAL) # Also allow saving the last frame when stopped
        self.status_var.set("Video processing stopped by user.")
         # Store the TOTAL accumulated counts up to the point of stopping
        self.detected_objects = self.detection_counts.copy()
        # Maybe show info about partial results
        messagebox.showinfo("Video Stopped", f"Video processing stopped.\nObjects detected up to this point (accumulated): {sum(self.detected_objects.values())}")


    def save_results(self, source_type):
        # Save the currently stored frame (last processed) and detection data
        if self.current_frame is None:
            messagebox.showinfo("Save Result", "No detection result (image or frame) available to save.")
            return

        # Get the output directory and create if it doesn't exist
        output_dir = self.output_dir_var.get()
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
             messagebox.showerror("Save Error", f"Could not create output directory:\n{output_dir}\nError: {e}")
             return

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{source_type}_result_{timestamp}"
        image_filename = os.path.join(output_dir, f"{base_filename}.jpg")
        data_filename = os.path.join(output_dir, f"{base_filename}_data.json")

        try:
            # --- Save image (the self.current_frame numpy array) ---
            success = cv2.imwrite(image_filename, self.current_frame)
            if not success:
                raise IOError(f"Failed to write image file to {image_filename}")

            # --- Prepare detection data ---
            # For 'image', self.detected_objects was set in detect_image
            # For 'video', self.detected_objects was set in on_video_complete/on_video_stop (total counts)
            detection_data = {
                "timestamp": datetime.now().isoformat(),
                "source_type": source_type,
                "input_file": self.image_path_var.get() if source_type == "image" else self.video_path_var.get(),
                "settings": {
                     "confidence_threshold": self.confidence_threshold,
                     "detected_classes": [self.class_names[idx] for idx, var in self.class_vars.items() if var.get()],
                 },
                "detection_counts": self.detected_objects, # Contains image counts or *total* video counts
                "total_objects": sum(self.detected_objects.values())
            }
            if source_type == "video":
                detection_data["info"] = "Counts represent total accumulated objects during processing."


            # --- Save detection data as JSON ---
            with open(data_filename, 'w') as f:
                json.dump(detection_data, f, indent=4)

            self.status_var.set(f"Results saved to {output_dir}")
            messagebox.showinfo("Save Complete", f"Detection result saved:\nImage: {image_filename}\nData: {data_filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results.\nError: {e}")
            self.status_var.set(f"Error saving results: {e}")

    def on_closing(self):
        # Gracefully handle closing the application
        if self.processing_video:
            if messagebox.askyesno("Confirm Exit", "Video processing is ongoing. Stop and exit?"):
                self.running = False # Signal thread to stop
                if self.video_thread and self.video_thread.is_alive():
                    try:
                        self.video_thread.join(timeout=1.0) # Wait briefly for thread
                    except Exception as e:
                         print(f"Error joining video thread: {e}")
                self.root.destroy()
            else:
                return # Don't close if user cancels
        else:
            self.running = False # Ensure flag is false even if no video thread
            # Add any other cleanup needed here
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    # Set the icon (optional, replace 'icon.ico' or 'icon.png')
    # try:
    #     # For Windows .ico
    #     # root.iconbitmap('icon.ico')
    #     # For cross-platform .png (requires Pillow)
    #     # icon = ImageTk.PhotoImage(file='icon.png')
    #     # root.iconphoto(True, icon)
    #     pass # Add your icon path here
    # except Exception as e:
    #     print(f"Could not load icon: {e}")

    app = YOLODetectionApp(root)
    root.mainloop()