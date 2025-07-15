# config.py
import os

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model and data paths
MODEL_PATH = os.path.join(
    BASE_DIR, "runs2", "detect", "traffic_detector", "weights", "best_2.pt"
)
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
COUNT_LINE_Y = 300

# Overlay display settings (compact)
OVERLAY_X, OVERLAY_Y = 10, 10
OVERLAY_WIDTH, OVERLAY_HEIGHT = 280, 250
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_HEIGHT = 18

# Vehicle classes mapping
CLASS_NAMES = {
    0: "ambulance",
    1: "auto",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "motorbike",
    6: "truck"
}

# Output directory for logs
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
