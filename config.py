# filename: config.py
import os

# --- AI Model Paths ---
MODEL_PATH = "hand_classifier_best.pth"
CLASS_NAMES_PATH = "class_names.txt"

# --- Image Processing Constants ---
IMG_SIZE = 128
ARENA_BBOX = (862, 134, 1346, 787) 
ELIXIR_OFFSET = {'x': 11, 'y': 144, 'width': 386, 'height': 38}
# --- Offsets for OCR from your FourCards.PNG anchor ---
# These are the relative positions of all OCR regions
OCR_OFFSETS = [
    {'name': 'Princess Tower (Bottom Left)', 'x_offset': 5, 'y_offset': -189, 'width': 39, 'height': 19},
    {'name': 'Princess Tower (Bottom Right)', 'x_offset': 302, 'y_offset': -189, 'width': 39, 'height': 19},
    {'name': 'Princess Tower (Top Left)', 'x_offset': 5, 'y_offset': -681, 'width': 39, 'height': 19},
    {'name': 'Princess Tower (Top Right)', 'x_offset': 302, 'y_offset': -681, 'width': 39, 'height': 19},
    {'name': 'Time', 'x_offset': 370, 'y_offset': -790, 'width': 74, 'height': 27},
    {'name': 'King Tower (Top)', 'x_offset': 158, 'y_offset': -799, 'width': 45, 'height': 19},
    {'name': 'King Tower (Bottom)', 'x_offset': 158, 'y_offset': -53, 'width': 45, 'height': 19},
]

# --- Card Offsets ---
# These are the relative positions of the 4 cards from the anchor
CARD_OFFSETS_WITH_SIZE = [
    (19, 26, 95, 116),  # Card 1
    (126, 25, 93, 117), # Card 2
    (232, 23, 94, 120), # Card 3
    (339, 25, 93, 119)  # Card 4
]

# --- Debugging ---
# Set to True to show a live window with the OCR and card boxes drawn.
DEBUG_VISUALS = True