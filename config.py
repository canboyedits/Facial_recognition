import os
from pathlib import Path

# Base directory
BASE_DIR = Path("/Users/yash/Desktop/Projects/Facial_Recognition")

# Path configurations
DATA_DIR = BASE_DIR / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
MODELS_DIR = BASE_DIR / "models"

# File paths
FACE_DB_PATH = MODELS_DIR / "face_db.pkl"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-face.pt"

# Create directories if missing
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)