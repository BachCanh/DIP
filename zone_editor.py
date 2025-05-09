# zone_editor.py
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import config
from parking_zone_detector import ParkingZoneDetector

class ZoneEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Zone Editor")
        self.root.geometry("1100x700")
        self.root.configure(bg=config.BG_COLOR)
        
        self.detector = ParkingZoneDetector()
        
        # State variables
        self.image_path = None
        self.current_image = None
        self.display_image = None
        self.current_zone_points = []
        self.current_zone_type = tk.StringVar(value="legal")
        self.zone_files = []
        self.selected_zone_file = tk.StringVar()
        self.editing_mode = True
        self.canvas_width = 800
        self.canvas_height = 600
        self._center_window(1100, 700)
        
        # Setup UI
        self._setup_ui()
        self._refresh_zone_files()

    def _center_window(self, width, height):
        """Center the window on the screen."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        
    def _setup_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background=config.BG_COLOR)
        style.configure("TLabel", background=config.BG_COLOR)
        style.configure("TButton", padding=6)
        style.configure("TLabelframe.Label", font=("Helvetica", 10, "bold"))
        style.configure("TLabelframe", padding=10)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas area
        self.canvas_frame = ttk.LabelFrame(left_panel, text="Zone Editor")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_width, height=self.canvas_height, 
                                bg="black", cursor="cross", highlightthickness=1, relief="ridge")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Controls below image
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        type_frame = ttk.LabelFrame(controls_frame, text="Zone Type")
        type_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Radiobutton(type_frame, text="Legal", variable=self.current_zone_type, value="legal").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Illegal", variable=self.current_zone_type, value="illegal").pack(side=tk.LEFT, padx=5)

        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Button(buttons_frame, text="Load Image", command=self._load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Complete Zone", command=self._complete_zone).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Clear Current", command=self._clear_current_zone).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Clear All Zones", command=self._clear_all_zones).pack(side=tk.LEFT, padx=5)

        # Right panel
        right_panel = ttk.LabelFrame(main_frame, text="Zone Management")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        file_frame = ttk.Frame(right_panel)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(file_frame, text="Zone Filename:").pack(anchor=tk.W, pady=(0, 5))
        self.filename_entry = ttk.Entry(file_frame, width=25)
        self.filename_entry.pack(fill=tk.X, pady=(0, 5))
        self.filename_entry.insert(0, "parking_zones.json")

        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Save Zones", command=self._save_zones).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Load Zones", command=self._load_zones_from_file).pack(side=tk.LEFT)

        files_frame = ttk.LabelFrame(right_panel, text="Saved Zone Files")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        self.files_listbox = tk.Listbox(files_frame, width=25, height=10)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=scrollbar.set)
        self.files_listbox.bind('<<ListboxSelect>>', self._on_file_select)

        stats_frame = ttk.LabelFrame(right_panel, text="Zone Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        self.stats_text = tk.Text(stats_frame, width=25, height=8, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._update_stats()

        help_frame = ttk.LabelFrame(right_panel, text="Instructions")
        help_frame.pack(fill=tk.X, padx=5, pady=5)
        help_text = (
            "1. Load an image of the parking area\n"
            "2. Select zone type (legal/illegal)\n"
            "3. Click to place zone points\n"
            "4. Click 'Complete Zone' when done\n"
            "5. Save zones when finished\n\n"
            "Green = Legal Parking Areas\n"
            "Red = Illegal Parking Areas"
        )
        ttk.Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=230).pack(padx=5, pady=5)

    
    def _load_image(self):
        """Load an image for zone definition"""
        file_path = filedialog.askopenfilename(
            filetypes=(("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*"))
        )
        if not file_path:
            return
            
        try:
            self.image_path = file_path
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("Error", f"Failed to read image: {file_path}")
                return
                
            # Resize image to fit canvas if needed
            self._update_canvas()
            
            # Clear current zone points
            self.current_zone_points = []
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
    
    def _update_canvas(self):
        """Update the canvas with the current image and zones"""
        if self.current_image is None:
            return
            
        # Create a copy of the image to draw on
        display_img = self.current_image.copy()
        
        # Draw existing zones and current zone points
        if self.editing_mode and self.current_zone_points:
            display_img = self.detector.draw_zone_editor(
                display_img, self.current_zone_points, self.current_zone_type.get()
            )
        else:
            display_img = self.detector.draw_zones(display_img)
        
        # Resize image to fit canvas if needed
        h, w = display_img.shape[:2]
        scale = min(self.canvas_width / w, self.canvas_height / h)
        
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        # Convert to PhotoImage
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.display_image = ImageTk.PhotoImage(image=Image.fromarray(display_img))
        
        # Update canvas
        self.canvas.config(width=display_img.shape[1], height=display_img.shape[0])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
    
    def _on_canvas_click(self, event):
        if self.current_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return

        # Get click coordinates
        x, y = event.x, event.y

        # Calculate scale factors
        original_h, original_w = self.current_image.shape[:2]
        displayed_w = self.display_image.width()
        displayed_h = self.display_image.height()
        scale_x = original_w / displayed_w
        scale_y = original_h / displayed_h

        # Convert to original coordinates
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)

        # Ensure within bounds
        original_x = min(max(0, original_x), original_w - 1)
        original_y = min(max(0, original_y), original_h - 1)

        # Add point to current zone
        self.current_zone_points.append([original_x, original_y])

        # Update display
        self._update_canvas()
        
    def _complete_zone(self):
        """Complete the current zone and add it to the detector"""
        if len(self.current_zone_points) < 3:
            messagebox.showwarning("Warning", "Need at least 3 points to define a zone.")
            return
            
        # Add zone to detector
        self.detector.add_zone(self.current_zone_points, self.current_zone_type.get())
        
        # Reset current zone points
        self.current_zone_points = []
        
        # Update statistics and display
        self._update_stats()
        self._update_canvas()
        
        messagebox.showinfo("Success", f"{self.current_zone_type.get().capitalize()} zone added.")
    
    def _clear_current_zone(self):
        """Clear the current zone points"""
        self.current_zone_points = []
        self._update_canvas()
    
    def _clear_all_zones(self):
        """Clear all defined zones"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all zones?"):
            self.detector.clear_zones()
            self._update_stats()
            self._update_canvas()
    
    def _save_zones(self):
        """Save current zones to a file"""
        filename = self.filename_entry.get()
        if not filename:
            messagebox.showwarning("Warning", "Please enter a filename.")
            return
            
        # Add .json extension if not present
        if not filename.endswith('.json'):
            filename += '.json'
            
        filepath = self.detector.save_zones(filename)
        self._refresh_zone_files()
        messagebox.showinfo("Success", f"Zones saved to {filepath}")
    
    def _load_zones_from_file(self):
        """Load zones from a file"""
        filename = self.filename_entry.get()
        if not filename:
            messagebox.showwarning("Warning", "Please enter a filename.")
            return
            
        # Add .json extension if not present
        if not filename.endswith('.json'):
            filename += '.json'
            
        if self.detector.load_zones(filename):
            self._update_stats()
            self._update_canvas()
            messagebox.showinfo("Success", f"Zones loaded from {filename}")
        else:
            messagebox.showerror("Error", f"Could not load zones from {filename}")
    
    def _refresh_zone_files(self):
        """Refresh the list of available zone files"""
        self.files_listbox.delete(0, tk.END)
        if os.path.exists(self.detector.config_dir):
            self.zone_files = [f for f in os.listdir(self.detector.config_dir) 
                              if f.endswith('.json')]
            for file in self.zone_files:
                self.files_listbox.insert(tk.END, file)
    
    def _on_file_select(self, event):
        """Handle selection of a zone file"""
        if not self.files_listbox.curselection():
            return
            
        index = self.files_listbox.curselection()[0]
        filename = self.files_listbox.get(index)
        self.filename_entry.delete(0, tk.END)
        self.filename_entry.insert(0, filename)
    
    def _update_stats(self):
        """Update the zone statistics display"""
        self.stats_text.delete(1.0, tk.END)
        
        stats = f"Legal Zones: {len(self.detector.legal_zones)}\n"
        stats += f"Illegal Zones: {len(self.detector.illegal_zones)}\n\n"
        
        if self.image_path:
            stats += f"Image: {os.path.basename(self.image_path)}\n"
        
        self.stats_text.insert(tk.END, stats)

if __name__ == "__main__":
    root = tk.Tk()
    app = ZoneEditorApp(root)
    root.mainloop()