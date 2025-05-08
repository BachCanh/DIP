# ui_components.py
import tkinter as tk
from tkinter import ttk, font as tkFont
from PIL import Image, ImageTk
import cv2
import config # Import config for styles and constants

# --- Styling ---
def setup_styles(root):
    style = ttk.Style(root)
    style.theme_use(config.THEME)

    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(size=10)
    label_font = tkFont.Font(family="Arial", size=10)
    button_font = tkFont.Font(family="Arial", size=10, weight="bold")
    title_font = tkFont.Font(family="Arial", size=16, weight="bold")
    banner_font = tkFont.Font(family="Arial", size=9) # Smaller banner text
    status_font = tkFont.Font(family="Arial", size=9)

    style.configure(".", font=default_font, background=config.BG_COLOR, foreground=config.TEXT_COLOR)
    style.configure("TFrame", background=config.BG_COLOR)
    style.configure("Main.TFrame", background=config.BG_COLOR)
    style.configure("TLabel", padding="5 5", font=label_font, background=config.BG_COLOR)
    style.configure("Status.TLabel", font=status_font, background="#d0d0d0", foreground="#333333")
    style.configure("Conf.TLabel", font=label_font, background=config.FRAME_BG)
    style.configure("TButton", padding="8 4", font=button_font, foreground=config.BUTTON_FG, background=config.ACCENT_COLOR)
    style.map("TButton", background=[('active', '#6faaf0'), ('pressed', '#3b6ec2')], foreground=[('active', 'white')])
    style.configure("TEntry", padding="5 5", font=label_font)
    style.configure("TLabelframe", padding="10 10", font=label_font, background=config.FRAME_BG, relief=tk.GROOVE, borderwidth=1)
    style.configure("TLabelframe.Label", font=label_font, background=config.FRAME_BG, foreground=config.TEXT_COLOR)
    style.configure("TNotebook", background=config.BG_COLOR, borderwidth=1)
    style.configure("TNotebook.Tab", padding=[10, 5], font=button_font, background="#d0d0d0", foreground="#555555")
    style.map("TNotebook.Tab", background=[("selected", config.ACCENT_COLOR)], foreground=[("selected", config.BUTTON_FG)])
    style.configure("TScale", background=config.FRAME_BG)
    style.configure("TProgressbar", thickness=15, background=config.ACCENT_COLOR, troughcolor='#d0d0d0')
    style.configure("TCheckbutton", background=config.FRAME_BG, font=label_font)
    style.configure("TCombobox", font=label_font, padding="5 5")

# --- Banner ---
def create_banner(parent):
    banner_bg = "#1e3c72"
    banner_fg = "white"
    banner_frame = tk.Frame(parent, bg=banner_bg)
    banner_frame.pack(fill=tk.X, pady=(0, 10))

    title_font = tkFont.Font(family="Arial", size=16, weight="bold")
    title_label = tk.Label(banner_frame, text=config.PROJECT_TITLE, font=title_font, fg=banner_fg, bg=banner_bg)
    title_label.pack(side=tk.LEFT, padx=30, pady=10, expand=True)

    team_frame = tk.Frame(banner_frame, bg=banner_bg)
    team_frame.pack(side=tk.RIGHT, padx=20, pady=5)

    banner_font = tkFont.Font(family="Arial", size=9)
    team_label = tk.Label(team_frame, text=config.TEAM_INFO, font=banner_font, fg=banner_fg, bg=banner_bg, justify=tk.RIGHT)
    team_label.pack()
    return banner_frame

# --- UI Helper ---
def display_image_on_label(cv2_image, label_widget, max_width=800, max_height=600):
    """Converts OpenCV image and displays it on a Tkinter Label."""
    if cv2_image is None:
        label_widget.configure(image=None)
        label_widget.image = None
        return

    try:
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Resize for display
        img_width, img_height = pil_img.size
        ratio = min(max_width / img_width, max_height / img_height) if img_width > 0 and img_height > 0 else 1
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # Only downscale
        if new_width < img_width or new_height < img_height:
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(pil_img)
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk # Keep reference
    except Exception as e:
        print(f"Error displaying image: {e}")
        label_widget.configure(image=None)
        label_widget.image = None

# --- Tab Creation ---
def setup_image_tab(parent, app_instance):
    tab = ttk.Frame(parent, padding="5 5 5 5", style="Main.TFrame")
    tab.columnconfigure(0, weight=1)
    tab.rowconfigure(1, weight=1)

    # File frame
    ff = ttk.LabelFrame(tab, text="Image Source", padding="10 10")
    ff.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))
    ff.columnconfigure(1, weight=1)
    ttk.Label(ff, text="Select Image:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
    ttk.Entry(ff, textvariable=app_instance.image_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(ff, text="Browse...", command=app_instance.browse_image).grid(row=0, column=2, padx=5, pady=5)
    ttk.Button(ff, text="Detect Objects", command=app_instance.detect_image).grid(row=0, column=3, padx=5, pady=5)
    ttk.Button(ff, text="Save Result", command=lambda: app_instance.save_results("image")).grid(row=0, column=4, padx=5, pady=5)

    # Display frame
    df = ttk.Frame(tab, style="Main.TFrame")
    df.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
    paned = ttk.PanedWindow(df, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(paned, style="Main.TFrame")
    app_instance.image_display_label = ttk.Label(left_frame, background='black', anchor=tk.CENTER) # Store ref in app
    app_instance.image_display_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    right_frame = ttk.LabelFrame(paned, text="Detection Results", padding="10 10")
    app_instance.img_result_text = tk.Text(right_frame, width=30, height=15, wrap=tk.WORD, font=("Arial", 9), relief=tk.SOLID, borderwidth=1) # Store ref
    app_instance.img_result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    paned.add(left_frame, weight=3)
    paned.add(right_frame, weight=1)
    return tab

def setup_video_tab(parent, app_instance):
    tab = ttk.Frame(parent, padding="5 5 5 5", style="Main.TFrame")
    tab.columnconfigure(0, weight=1)
    tab.rowconfigure(2, weight=1)

    # File frame
    ff = ttk.LabelFrame(tab, text="Video Source", padding="10 10")
    ff.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))
    ff.columnconfigure(1, weight=1)
    ttk.Label(ff, text="Select Video:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
    ttk.Entry(ff, textvariable=app_instance.video_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(ff, text="Browse...", command=app_instance.browse_video).grid(row=0, column=2, padx=5, pady=5)

    # Control frame
    cf = ttk.Frame(tab, padding="5 0", style="Main.TFrame")
    cf.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 10))
    cf.columnconfigure(3, weight=1) # Progress bar expansion
    app_instance.detect_video_btn = ttk.Button(cf, text="Start Detection", command=app_instance.start_video_detection) # Store ref
    app_instance.detect_video_btn.grid(row=0, column=0, padx=(0, 5), pady=5)
    app_instance.stop_video_btn = ttk.Button(cf, text="Stop Detection", command=app_instance.stop_video_detection, state=tk.DISABLED) # Store ref
    app_instance.stop_video_btn.grid(row=0, column=1, padx=5, pady=5)
    app_instance.save_video_btn = ttk.Button(cf, text="Save Last Frame", command=lambda: app_instance.save_results("video")) # Store ref
    app_instance.save_video_btn.grid(row=0, column=2, padx=5, pady=5)
    app_instance.video_progress = ttk.Progressbar(cf, length=250, mode='determinate', style="TProgressbar") # Store ref
    app_instance.video_progress.grid(row=0, column=3, padx=10, pady=5, sticky="ew")

    # Display frame
    df = ttk.Frame(tab, style="Main.TFrame")
    df.grid(row=2, column=0, sticky="nsew", padx=5, pady=(0, 5))
    paned = ttk.PanedWindow(df, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(paned, style="Main.TFrame")
    app_instance.video_display_label = ttk.Label(left_frame, background='black', anchor=tk.CENTER) # Store ref
    app_instance.video_display_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    right_frame = ttk.LabelFrame(paned, text="Live Frame Stats", padding="10 10")
    app_instance.video_stats_text = tk.Text(right_frame, width=30, height=15, wrap=tk.WORD, font=("Arial", 9), relief=tk.SOLID, borderwidth=1) # Store ref
    app_instance.video_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    paned.add(left_frame, weight=3)
    paned.add(right_frame, weight=1)
    return tab


def setup_settings_tab(parent, app_instance):
    # Using a Canvas for scrolling content if needed, simplified here
    tab = ttk.Frame(parent, padding="10 10 10 10", style="Main.TFrame")

    # --- Detection Settings ---
    settings_frame = ttk.LabelFrame(tab, text="Detection Settings", padding="15 15")
    settings_frame.pack(fill=tk.X, padx=5, pady=5, anchor=tk.N)
    settings_frame.columnconfigure(1, weight=1)

    ttk.Label(settings_frame, text="Confidence:").grid(row=0, column=0, padx=5, pady=8, sticky=tk.W)
    app_instance.conf_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, length=250,
                                         orient=tk.HORIZONTAL, command=app_instance.update_conf_display) # Store ref
    app_instance.conf_scale.grid(row=0, column=1, padx=5, pady=8, sticky=tk.EW)
    app_instance.conf_label = ttk.Label(settings_frame, text="", style="Conf.TLabel", width=5) # Store ref
    app_instance.conf_label.grid(row=0, column=2, padx=5, pady=8)

    # --- Class selection ---
    ttk.Label(settings_frame, text="Filter Classes:").grid(row=1, column=0, padx=5, pady=8, sticky=tk.NW)
    class_frame = ttk.Frame(settings_frame, style="TLabelframe") # Use frame background
    class_frame.grid(row=1, column=1, columnspan=2, padx=5, pady=8, sticky=tk.W)

    app_instance.class_vars = {} # Store class checkbutton vars here
    num_cols = 3
    row, col = 0, 0
    if app_instance.detector and app_instance.detector.class_names:
         for idx, name in app_instance.detector.class_names.items():
             app_instance.class_vars[idx] = tk.BooleanVar(value=True)
             cb = ttk.Checkbutton(class_frame, text=name.capitalize(), variable=app_instance.class_vars[idx])
             cb.grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
             col += 1
             if col >= num_cols:
                 col = 0; row += 1
    else:
        ttk.Label(class_frame, text="Model/Classes not loaded.").grid(row=0, column=0)


    # --- Output directory ---
    output_frame = ttk.LabelFrame(tab, text="Output Settings", padding="15 15")
    output_frame.pack(fill=tk.X, padx=5, pady=10, anchor=tk.N)
    output_frame.columnconfigure(1, weight=1)
    ttk.Label(output_frame, text="Output Dir:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    app_instance.output_dir_var = tk.StringVar() # Store ref
    ttk.Entry(output_frame, textvariable=app_instance.output_dir_var, width=40).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(output_frame, text="Browse...", command=app_instance.browse_output_dir).grid(row=0, column=2, padx=5, pady=5)

    # --- Action Buttons ---
    button_frame = ttk.Frame(tab, style="Main.TFrame")
    button_frame.pack(fill=tk.X, padx=5, pady=15)
    ttk.Button(button_frame, text="Apply Settings", command=app_instance.apply_settings).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Save Settings", command=app_instance.save_settings).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Load Settings", command=app_instance.load_settings).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Reset Defaults", command=app_instance.reset_settings).pack(side=tk.LEFT, padx=10)

    # --- About section (Optional, simplified) ---
    about_frame = ttk.LabelFrame(tab, text="About", padding="10 10")
    about_frame.pack(fill=tk.X, padx=5, pady=10, anchor=tk.N)
    about_text = "YOLO Detection App v1.1 (Refactored)\nUsing Ultralytics YOLOv8"
    ttk.Label(about_frame, text=about_text, justify=tk.LEFT, style="TLabelframe.Label").pack(anchor=tk.W, padx=5, pady=5)

    return tab

# --- Status Bar ---
def create_status_bar(parent, status_var):
    status_bar = ttk.Label(parent, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5 2", style="Status.TLabel")
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    return status_bar