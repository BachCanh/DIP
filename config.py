# config.py
import os

# --- Model ---
# IMPORTANT: Adjust this path relative to where main.py is run
MODEL_PATH = "runs/train_cars_20250422_090512/yolov8n/weights/best.pt"

# --- Settings ---
SETTINGS_DIR = "./config"
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")
DEFAULT_CONFIDENCE = 0.25
DEFAULT_OUTPUT_DIR = "./output"

# --- UI ---
WINDOW_WIDTH = 1200 # Slightly reduced for potentially tighter layout
WINDOW_HEIGHT = 800
THEME = 'clam'
BG_COLOR = "#f0f0f0"
FRAME_BG = "#e0e0e0"
ACCENT_COLOR = "#4a86e8"
TEXT_COLOR = "#333333"
BUTTON_FG = "white"

# --- Team Info ---
TEAM_INFO = (
    "Team:\n"
    "Bạch Đức Cảnh (22110012) | Nguyễn Tiến Toàn (22110078)\n"
    "Lý Đăng Triều (22110080) | Nguyễn Tuấn Vũ (22110091)"
)
PROJECT_TITLE = "Car Detection - Final Project"