"""
AuraScan - Main Application
Real-time heart rate monitoring using computer vision and signal processing
"""

import cv2
import numpy as np
import sys

from face_detector import FaceDetector
from signal_processor import SignalProcessor
import config


def draw_waveform(frame: np.ndarray, waveform_data: list) -> np.ndarray:
    """
    Draw a real-time ECG-style waveform on the frame.
    
    Args:
        frame: BGR image to draw on
        waveform_data: List of filtered signal values
        
    Returns:
        Frame with waveform overlay
    """
    if not waveform_data or len(waveform_data) < 2:
        return frame
    
    h, w = frame.shape[:2]
    
    # Waveform panel dimensions
    panel_height = config.WAVEFORM_HEIGHT
    panel_width = w - 40
    panel_x = 20
    panel_y = h - panel_height - 20
    
    # Create semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x - 10, panel_y - 10),
        (panel_x + panel_width + 10, panel_y + panel_height + 10),
        (20, 20, 20),
        -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw grid lines for medical monitor effect
    grid_color = (40, 40, 40)
    # Horizontal grid lines
    for i in range(5):
        y = panel_y + int((panel_height / 4) * i)
        cv2.line(frame, (panel_x, y), (panel_x + panel_width, y), grid_color, 1)
    
    # Vertical grid lines
    for i in range(10):
        x = panel_x + int((panel_width / 9) * i)
        cv2.line(frame, (x, panel_y), (x, panel_y + panel_height), grid_color, 1)
    
    # Draw border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (60, 60, 60),
        2
    )
    
    # Normalize waveform data to fit panel height
    waveform_array = np.array(waveform_data, dtype=np.float32)
    
    # Handle edge case: constant signal
    if np.std(waveform_array) < 1e-10:
        # Draw flat line at center
        center_y = panel_y + panel_height // 2
        cv2.line(
            frame,
            (panel_x, center_y),
            (panel_x + panel_width, center_y),
            config.WAVEFORM_COLOR,
            2
        )
        return frame
    
    # Normalize to [-1, 1] range
    waveform_normalized = (waveform_array - np.mean(waveform_array)) / (np.std(waveform_array) * 3)
    waveform_normalized = np.clip(waveform_normalized, -1, 1)
    
    # Convert to pixel coordinates
    # Map [-1, 1] to [panel_y + panel_height, panel_y] (inverted Y axis)
    waveform_y = panel_y + panel_height // 2 - (waveform_normalized * (panel_height // 2 - 10)).astype(int)
    
    # Calculate X coordinates
    num_points = len(waveform_data)
    waveform_x = panel_x + np.linspace(0, panel_width, num_points).astype(int)
    
    # Draw the waveform line
    points = np.column_stack((waveform_x, waveform_y))
    
    # Draw line segments with glow effect
    for i in range(len(points) - 1):
        # Outer glow (thicker, semi-transparent)
        cv2.line(
            frame,
            tuple(points[i]),
            tuple(points[i + 1]),
            config.WAVEFORM_COLOR,
            3,
            cv2.LINE_AA
        )
    
    # Draw brighter core line
    for i in range(len(points) - 1):
        cv2.line(
            frame,
            tuple(points[i]),
            tuple(points[i + 1]),
            (100, 255, 150),
            1,
            cv2.LINE_AA
        )
    
    # Add label
    cv2.putText(
        frame,
        "PULSE WAVEFORM",
        (panel_x, panel_y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
        cv2.LINE_AA
    )
    
    return frame


def extract_roi_pixels(frame: np.ndarray, roi_points: list) -> np.ndarray:
    """
    Extract pixel values from the ROI using a mask.
    
    Args:
        frame: BGR image from webcam
        roi_points: list of (x, y) tuples defining the ROI polygon
        
    Returns:
        Array of BGR pixels within the ROI region (N x 3 array)
    """
    # Create a mask for the ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Convert roi_points to numpy array
    points = np.array(roi_points, dtype=np.int32)
    
    # Fill the polygon on the mask
    cv2.fillPoly(mask, [points], 255)
    
    # Create a 3-channel mask
    mask_3channel = cv2.merge([mask, mask, mask])
    
    # Apply mask to frame
    masked_frame = cv2.bitwise_and(frame, mask_3channel)
    
    # Extract only the pixels inside the ROI (where mask is white)
    roi_pixels = masked_frame[mask == 255]
    
    return roi_pixels


def draw_ui_overlay(frame: np.ndarray, bpm: float, buffer_ready: bool, buffer_size: int) -> np.ndarray:
    """
    Draw enhanced UI overlay with BPM, status, and instructions on the frame.
    
    Args:
        frame: BGR image to draw on
        bpm: Current BPM value (None if not available)
        buffer_ready: Whether the signal buffer is full
        buffer_size: Current buffer fill level
        
    Returns:
        Frame with UI overlay
    """
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay panel at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    # Add decorative corner brackets for sci-fi effect
    corner_color = (0, 200, 255)
    corner_len = 30
    
    # Top-left corner
    cv2.line(frame, (10, 10), (10 + corner_len, 10), corner_color, 2)
    cv2.line(frame, (10, 10), (10, 10 + corner_len), corner_color, 2)
    
    # Top-right corner
    cv2.line(frame, (w - 10, 10), (w - 10 - corner_len, 10), corner_color, 2)
    cv2.line(frame, (w - 10, 10), (w - 10, 10 + corner_len), corner_color, 2)
    
    # Display BPM or status
    if bpm is not None:
        # Main BPM display
        bpm_text = f"{bpm:.1f}"
        color = (0, 255, 100)  # Bright green when BPM is available
        
        # Draw BPM value large and prominent
        cv2.putText(
            frame,
            bpm_text,
            (20, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            color,
            3,
            cv2.LINE_AA
        )
        
        # Add "BPM" label
        cv2.putText(
            frame,
            "BPM",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )
        
        # Add heart icon simulation using circle
        heart_x = int(20 + len(bpm_text) * 25 + 40)
        cv2.circle(frame, (heart_x, 50), 15, color, -1)
        cv2.putText(
            frame,
            "♥",
            (heart_x - 10, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (20, 20, 20),
            2,
            cv2.LINE_AA
        )
        
        # Status indicator
        cv2.putText(
            frame,
            "● MONITORING ACTIVE",
            (w - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA
        )
    else:
        # Show acquisition status
        if buffer_ready:
            status_text = "ANALYZING SIGNAL..."
            color = (0, 165, 255)  # Orange
        else:
            progress = int((buffer_size / config.BUFFER_SIZE) * 100)
            status_text = f"ACQUIRING SIGNAL: {progress}%"
            color = (0, 165, 255)  # Orange
        
        cv2.putText(
            frame,
            status_text,
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )
        
        # Progress bar
        bar_width = 300
        bar_height = 8
        bar_x = 20
        bar_y = 70
        
        # Background bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (60, 60, 60),
            -1
        )
        
        # Progress fill
        if buffer_size > 0:
            progress_width = int((buffer_size / config.BUFFER_SIZE) * bar_width)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                color,
                -1
            )
    
    # Instructions
    cv2.putText(
        frame,
        "Stay still | Good lighting | Press 'Q' to quit | 'R' to reset",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )
    
    return frame


def main():
    """
    Main application loop for AuraScan heart rate monitoring.
    """
    print("=" * 60)
    print("AuraScan - Remote Vitals Monitoring")
    print("=" * 60)
    print("\n📋 Instructions:")
    print("  • Ensure good, even lighting on your face")
    print("  • Avoid backlighting (windows behind you)")
    print("  • Stay as still as possible")
    print("  • Wait ~5 seconds for initial reading")
    print("  • Press 'Q' to quit\n")
    print("Starting webcam...\n")
    
    # Initialize components
    try:
        webcam = cv2.VideoCapture(config.WEBCAM_ID)
        
        if not webcam.isOpened():
            print("❌ Error: Could not access webcam")
            sys.exit(1)
        
        # Set webcam properties for better performance
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        webcam.set(cv2.CAP_PROP_FPS, config.FPS)
        
        face_detector = FaceDetector()
        signal_processor = SignalProcessor()
        
        print("✅ System initialized successfully")
        print("🎥 Webcam active - Starting monitoring...\n")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        sys.exit(1)
    
    # Main processing loop
    frame_count = 0
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = webcam.read()
            
            if not ret:
                print("⚠️  Warning: Failed to read frame from webcam")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)
            
            # Detect face and get ROI
            roi_points, landmarks = face_detector.detect(frame)
            
            current_bpm = None
            
            if roi_points is not None:
                # Extract pixels from the forehead ROI
                roi_pixels = extract_roi_pixels(frame, roi_points)
                
                # Process the ROI pixels to calculate BPM
                current_bpm = signal_processor.process_frame(roi_pixels)
                
                # Draw the ROI on the frame for visual feedback
                frame = face_detector.draw_roi(frame, roi_points)
            else:
                # No face detected - show warning
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Draw UI overlay with BPM and status
            frame = draw_ui_overlay(
                frame,
                current_bpm,
                signal_processor.is_buffer_ready(),
                signal_processor.get_buffer_size()
            )
            
            # Draw real-time waveform if we have data
            waveform_data = signal_processor.get_waveform_data()
            if len(waveform_data) > 0:
                frame = draw_waveform(frame, waveform_data)
            
            # Display the frame
            cv2.imshow('AuraScan - Heart Rate Monitor', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Shutting down...")
                break
            elif key == ord('r') or key == ord('R'):
                # Reset the signal processor
                signal_processor.reset()
                print("🔄 Signal processor reset")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
    
    finally:
        # Clean up resources
        webcam.release()
        cv2.destroyAllWindows()
        print("✅ Resources released")
        print("=" * 60)


if __name__ == "__main__":
    main()
