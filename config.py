"""
AuraScan Configuration File
Contains constants for signal processing and UI settings
"""

import cv2

# Buffer settings
BUFFER_SIZE = 150  # Number of frames to keep for analysis

# Frame rate settings
FPS = 30  # Assumed webcam frame rate

# Bandpass filter settings (in Hz)
LOW_CUT = 0.7  # Corresponds to ~42 BPM
HIGH_CUT = 4.0  # Corresponds to ~240 BPM

# UI settings
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # Consistent font for UI elements

# Waveform visualization settings
WAVEFORM_WINDOW_SIZE = 150  # Number of points to display on the graph
WAVEFORM_HEIGHT = 120  # Height of the waveform panel in pixels
WAVEFORM_COLOR = (0, 255, 100)  # Bright green for ECG-style line
