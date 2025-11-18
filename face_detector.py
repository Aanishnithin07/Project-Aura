"""
Face Detector Module
Uses MediaPipe Face Mesh to detect faces and extract forehead ROI for vital signs monitoring
"""

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    Detects faces and extracts forehead Region of Interest (ROI) using MediaPipe Face Mesh.
    """
    
    # Optimized forehead landmark indices - focused on flatter central region to reduce noise
    FOREHEAD_LANDMARKS = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    def __init__(self):
        """
        Initialize the FaceDetector with MediaPipe Face Mesh.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect(self, frame):
        """
        Detect face and extract forehead ROI coordinates.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            tuple: (roi_coordinates, landmarks) if face detected, else (None, None)
                   roi_coordinates: list of (x, y) tuples for forehead region
                   landmarks: raw MediaPipe landmarks object
        """
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        # Check if face is detected
        if not results.multi_face_landmarks:
            return None, None
        
        # Get the first (and only) face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract forehead ROI coordinates
        h, w, _ = frame.shape
        roi_coordinates = []
        
        for landmark_idx in self.FOREHEAD_LANDMARKS:
            landmark = face_landmarks.landmark[landmark_idx]
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            roi_coordinates.append((x, y))
        
        return roi_coordinates, face_landmarks
    
    def draw_roi(self, frame, roi_points):
        """
        Draw a green polygon around the detected forehead ROI for debugging.
        
        Args:
            frame: BGR image to draw on
            roi_points: list of (x, y) tuples defining the ROI polygon
            
        Returns:
            frame: Image with ROI polygon drawn
        """
        if roi_points is None or len(roi_points) == 0:
            return frame
        
        # Get frame dimensions for boundary checking
        h, w = frame.shape[:2]
        
        # Convert roi_points to numpy array and clip to frame bounds
        points = np.array(roi_points, dtype=np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)  # Clip x coordinates
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)  # Clip y coordinates
        
        # Draw the polygon
        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=(0, 255, 0),  # Green color
            thickness=2
        )
        
        # Optionally fill the polygon with semi-transparent green
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return frame
    
    def __del__(self):
        """
        Clean up resources when the detector is destroyed.
        """
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
