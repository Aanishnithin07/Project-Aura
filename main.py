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
    Draw UI overlay with BPM, status, and instructions on the frame.
    
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
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    # Display BPM or status
    if bpm is not None:
        bpm_text = f"Heart Rate: {bpm:.1f} BPM"
        color = (0, 255, 0)  # Green when BPM is available
        cv2.putText(
            frame,
            bpm_text,
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            color,
            2,
            cv2.LINE_AA
        )
    else:
        # Show acquisition status
        if buffer_ready:
            status_text = "Signal: Processing..."
            color = (0, 165, 255)  # Orange
        else:
            progress = int((buffer_size / config.BUFFER_SIZE) * 100)
            status_text = f"Signal: Acquiring... {progress}%"
            color = (0, 165, 255)  # Orange
        
        cv2.putText(
            frame,
            status_text,
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            color,
            2,
            cv2.LINE_AA
        )
    
    # Instructions
    cv2.putText(
        frame,
        "Stay still | Good lighting | Press 'Q' to quit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
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
