"""
AuraScan Configuration File
Contains constants for signal processing and webcam settings
"""

# Buffer settings
BUFFER_SIZE = 150  # Number of frames to keep for analysis

# Webcam settings
FPS = 30  # Assumed webcam frame rate
WEBCAM_ID = 0  # Default webcam

# Bandpass filter settings (in Hz)
LOW_CUT = 0.7  # Corresponds to ~42 BPM
HIGH_CUT = 4.0  # Corresponds to ~240 BPM
