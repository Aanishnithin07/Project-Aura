"""
UI Renderer Module - Professional HUD-style interface for AuraScan
Handles all visual elements including waveform graphs and status panels
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class UIRenderer:
    """
    Renders a futuristic heads-up display (HUD) interface for real-time heart rate monitoring.
    Provides ECG-style waveform graphs and status panels with sci-fi aesthetics.
    """
    
    def __init__(
        self,
        graph_width: int = 400,
        graph_height: int = 120,
        graph_color: Tuple[int, int, int] = (0, 255, 0)
    ):
        """
        Initialize the UI renderer with customizable dimensions and colors.
        
        Args:
            graph_width: Width of the waveform graph box in pixels (default: 400)
            graph_height: Height of the waveform graph box in pixels (default: 120)
            graph_color: BGR color for the waveform line (default: green)
        """
        self.graph_width = graph_width
        self.graph_height = graph_height
        self.graph_color = graph_color
        
        # UI styling constants
        self.panel_bg_color = (20, 20, 20)  # Dark background
        self.panel_border_color = (60, 60, 60)  # Border color
        self.grid_color = (40, 40, 40)  # Grid lines
        self.text_color = (200, 200, 200)  # Light gray text
        self.accent_color = (0, 200, 255)  # Cyan accent
        self.warning_color = (0, 0, 255)  # Red for warnings
        self.success_color = (0, 255, 100)  # Green for success
    
    def draw_graph(
        self,
        frame: np.ndarray,
        signal_data: np.ndarray,
        top_left_xy: Tuple[int, int]
    ) -> np.ndarray:
        """
        Draw a real-time ECG-style waveform graph with scrolling effect.
        
        Args:
            frame: BGR image to draw on
            signal_data: Array of normalized signal values to plot
            top_left_xy: (x, y) tuple for top-left corner of graph
            
        Returns:
            Frame with waveform graph overlay
        """
        if signal_data is None or len(signal_data) < 2:
            return frame
        
        x_start, y_start = top_left_xy
        
        # Create semi-transparent background panel
        overlay = frame.copy()
        
        # Draw background rectangle with padding
        padding = 10
        cv2.rectangle(
            overlay,
            (x_start - padding, y_start - padding),
            (x_start + self.graph_width + padding, y_start + self.graph_height + padding),
            self.panel_bg_color,
            -1
        )
        
        # Blend overlay with frame for transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw grid lines for medical monitor effect
        self._draw_grid(frame, x_start, y_start)
        
        # Draw border
        cv2.rectangle(
            frame,
            (x_start, y_start),
            (x_start + self.graph_width, y_start + self.graph_height),
            self.panel_border_color,
            2
        )
        
        # Normalize signal data to fit graph height
        normalized_signal = self._normalize_signal(signal_data)
        
        # Convert normalized values to pixel coordinates
        num_points = len(normalized_signal)
        
        # Calculate X coordinates (spread evenly across width)
        x_coords = np.linspace(x_start, x_start + self.graph_width, num_points).astype(int)
        
        # Calculate Y coordinates (inverted because Y increases downward)
        # Map [-1, 1] to [y_start + height, y_start] with padding
        y_padding = 15
        y_coords = (
            y_start + self.graph_height // 2 -
            (normalized_signal * (self.graph_height // 2 - y_padding))
        ).astype(int)
        
        # Draw the waveform with glow effect
        points = np.column_stack((x_coords, y_coords))
        
        # Outer glow (thicker line)
        for i in range(len(points) - 1):
            cv2.line(
                frame,
                tuple(points[i]),
                tuple(points[i + 1]),
                self.graph_color,
                3,
                cv2.LINE_AA
            )
        
        # Inner bright core
        for i in range(len(points) - 1):
            cv2.line(
                frame,
                tuple(points[i]),
                tuple(points[i + 1]),
                (100, 255, 150),  # Brighter core
                1,
                cv2.LINE_AA
            )
        
        # Add graph label
        cv2.putText(
            frame,
            "PULSE WAVEFORM",
            (x_start, y_start - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
            cv2.LINE_AA
        )
        
        return frame
    
    def draw_info_panel(
        self,
        frame: np.ndarray,
        bpm: Optional[float],
        face_detected: bool
    ) -> np.ndarray:
        """
        Draw a futuristic information panel with BPM and status indicators.
        
        Args:
            frame: BGR image to draw on
            bpm: Current heart rate in BPM (None if not available)
            face_detected: Whether a face is currently detected
            
        Returns:
            Frame with info panel overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent top panel
        overlay = frame.copy()
        panel_height = 140
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)
        
        # Draw corner brackets for tactical HUD effect
        self._draw_corner_brackets(frame, w)
        
        if not face_detected:
            # Target searching mode
            self._draw_searching_status(frame, w)
        else:
            # Target locked mode
            self._draw_locked_status(frame, w, bpm)
        
        # Draw instructions at bottom of panel
        cv2.putText(
            frame,
            "CONTROLS: [Q] Quit | [R] Reset | Stay still for best results",
            (20, panel_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
            cv2.LINE_AA
        )
        
        return frame
    
    def _draw_grid(self, frame: np.ndarray, x_start: int, y_start: int) -> None:
        """
        Draw grid lines on the waveform graph for medical monitor aesthetic.
        
        Args:
            frame: Frame to draw on
            x_start: X coordinate of graph start
            y_start: Y coordinate of graph start
        """
        # Horizontal grid lines (5 lines)
        for i in range(5):
            y = y_start + int((self.graph_height / 4) * i)
            cv2.line(
                frame,
                (x_start, y),
                (x_start + self.graph_width, y),
                self.grid_color,
                1
            )
        
        # Vertical grid lines (10 lines)
        for i in range(10):
            x = x_start + int((self.graph_width / 9) * i)
            cv2.line(
                frame,
                (x, y_start),
                (x, y_start + self.graph_height),
                self.grid_color,
                1
            )
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal data to [-1, 1] range for consistent visualization.
        
        Args:
            signal: Raw signal data
            
        Returns:
            Normalized signal in [-1, 1] range
        """
        signal_array = np.array(signal, dtype=np.float32)
        
        # Handle constant signal
        if np.std(signal_array) < 1e-10:
            return np.zeros_like(signal_array)
        
        # Normalize to [-1, 1] with 3 standard deviations
        normalized = (signal_array - np.mean(signal_array)) / (np.std(signal_array) * 3)
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def _draw_corner_brackets(self, frame: np.ndarray, width: int) -> None:
        """
        Draw decorative corner brackets for sci-fi HUD effect.
        
        Args:
            frame: Frame to draw on
            width: Frame width
        """
        corner_len = 30
        
        # Top-left corner
        cv2.line(frame, (10, 10), (10 + corner_len, 10), self.accent_color, 2)
        cv2.line(frame, (10, 10), (10, 10 + corner_len), self.accent_color, 2)
        
        # Top-right corner
        cv2.line(frame, (width - 10, 10), (width - 10 - corner_len, 10), self.accent_color, 2)
        cv2.line(frame, (width - 10, 10), (width - 10, 10 + corner_len), self.accent_color, 2)
    
    def _draw_searching_status(self, frame: np.ndarray, width: int) -> None:
        """
        Draw searching/scanning status when no face is detected.
        
        Args:
            frame: Frame to draw on
            width: Frame width
        """
        # Main status text
        cv2.putText(
            frame,
            "SEARCHING TARGET...",
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            self.warning_color,
            2,
            cv2.LINE_AA
        )
        
        # Animated scanning indicator
        cv2.putText(
            frame,
            "⚠ NO FACE DETECTED",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.warning_color,
            1,
            cv2.LINE_AA
        )
        
        # Status indicator (right side)
        cv2.putText(
            frame,
            "STATUS: SCANNING",
            (width - 220, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.warning_color,
            1,
            cv2.LINE_AA
        )
    
    def _draw_locked_status(
        self,
        frame: np.ndarray,
        width: int,
        bpm: Optional[float]
    ) -> None:
        """
        Draw locked status with BPM display when face is detected.
        
        Args:
            frame: Frame to draw on
            width: Frame width
            bpm: Current BPM value
        """
        if bpm is not None:
            # Large BPM display
            bpm_text = f"{bpm:.1f}"
            cv2.putText(
                frame,
                bpm_text,
                (20, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                2.0,
                self.success_color,
                3,
                cv2.LINE_AA
            )
            
            # BPM label
            cv2.putText(
                frame,
                "BPM",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.text_color,
                1,
                cv2.LINE_AA
            )
            
            # Heart icon
            heart_x = int(20 + len(bpm_text) * 25 + 40)
            cv2.circle(frame, (heart_x, 50), 15, self.success_color, -1)
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
                "● STATUS: LOCKED",
                (width - 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.success_color,
                1,
                cv2.LINE_AA
            )
        else:
            # Acquiring signal
            cv2.putText(
                frame,
                "ACQUIRING SIGNAL...",
                (20, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                self.accent_color,
                2,
                cv2.LINE_AA
            )
            
            # Status indicator
            cv2.putText(
                frame,
                "○ STATUS: LOCKED",
                (width - 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.accent_color,
                1,
                cv2.LINE_AA
            )
    
    def draw_progress_bar(
        self,
        frame: np.ndarray,
        progress: float,
        position: Tuple[int, int],
        width: int = 300,
        height: int = 8
    ) -> np.ndarray:
        """
        Draw a progress bar for signal acquisition.
        
        Args:
            frame: Frame to draw on
            progress: Progress value from 0.0 to 1.0
            position: (x, y) tuple for bar position
            width: Bar width in pixels
            height: Bar height in pixels
            
        Returns:
            Frame with progress bar
        """
        x, y = position
        
        # Background bar
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (60, 60, 60),
            -1
        )
        
        # Progress fill
        progress_width = int(progress * width)
        if progress_width > 0:
            cv2.rectangle(
                frame,
                (x, y),
                (x + progress_width, y + height),
                self.accent_color,
                -1
            )
        
        # Border
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            self.panel_border_color,
            1
        )
        
        return frame
