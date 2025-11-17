"""
Signal Processor Module
Handles raw color signal data and converts it to heart rate using advanced DSP techniques
"""

from collections import deque
from typing import Optional, List
import numpy as np
from scipy import signal

import config


class SignalProcessor:
    """
    Processes raw color signals from forehead ROI and calculates heart rate (BPM).
    Uses detrending, normalization, bandpass filtering, and FFT for robust BPM estimation.
    """
    
    def __init__(self):
        """
        Initialize the SignalProcessor with an empty buffer.
        """
        # Use deque for efficient append and automatic size limiting
        self.buffer: deque = deque(maxlen=config.BUFFER_SIZE)
        self.current_bpm: Optional[float] = None
        
        # Waveform visualization buffer - stores filtered signal for display
        self.waveform_buffer: deque = deque(maxlen=config.WAVEFORM_WINDOW_SIZE)
        self.filtered_signal_value: Optional[float] = None
        
        # Pre-calculate Butterworth filter coefficients for efficiency
        self._setup_bandpass_filter()
    
    def _setup_bandpass_filter(self) -> None:
        """
        Pre-calculate Butterworth bandpass filter coefficients.
        """
        # Nyquist frequency (half the sampling rate)
        nyquist = 0.5 * config.FPS
        
        # Normalize cutoff frequencies
        low = config.LOW_CUT / nyquist
        high = config.HIGH_CUT / nyquist
        
        # Design 4th-order Butterworth bandpass filter
        self.filter_b, self.filter_a = signal.butter(
            N=4,
            Wn=[low, high],
            btype='bandpass',
            analog=False
        )
    
    def process_frame(self, roi_pixels: np.ndarray) -> Optional[float]:
        """
        Process a single frame's ROI pixels by extracting the green channel average.
        
        Args:
            roi_pixels: numpy array of BGR pixels from the forehead ROI
            
        Returns:
            Current BPM if buffer is full and calculation succeeds, else None
        """
        if roi_pixels is None or len(roi_pixels) == 0:
            return None
        
        try:
            # Extract GREEN channel (index 1 in BGR format)
            # Green light is most sensitive to blood volume changes
            # roi_pixels is a (N, 3) array where N is number of pixels and 3 is BGR
            green_channel = roi_pixels[:, 1]
            
            # Calculate mean green value for this frame
            mean_green = np.mean(green_channel)
            
            # Add to buffer
            self.buffer.append(mean_green)
            
            # Calculate BPM only when buffer is full
            if len(self.buffer) == config.BUFFER_SIZE:
                return self.calculate_bpm()
            
            return None
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def calculate_bpm(self) -> Optional[float]:
        """
        Calculate heart rate from the buffered signal using DSP techniques.
        
        Pipeline:
        1. Detrending - Remove slow baseline drift
        2. Normalization - Standardize signal amplitude
        3. Bandpass Filtering - Remove noise outside heart rate range
        4. FFT - Find dominant frequency
        5. Frequency to BPM conversion
        
        Returns:
            Estimated BPM value, or None if calculation fails
        """
        if len(self.buffer) < config.BUFFER_SIZE:
            return None
        
        try:
            # Convert buffer to numpy array
            raw_signal = np.array(self.buffer, dtype=np.float64)
            
            # Step 1: Detrending - Remove polynomial trend (linear by default)
            detrended_signal = signal.detrend(raw_signal, type='linear')
            
            # Step 2: Normalization - Zero mean, unit variance
            mean_val = np.mean(detrended_signal)
            std_val = np.std(detrended_signal)
            
            # Handle edge case: constant signal (std = 0)
            if std_val < 1e-10:
                return self.current_bpm
            
            normalized_signal = (detrended_signal - mean_val) / std_val
            
            # Step 3: Bandpass Filtering - Keep only heart rate frequencies
            # Using filtfilt for zero-phase filtering (no time delay)
            filtered_signal = signal.filtfilt(
                self.filter_b,
                self.filter_a,
                normalized_signal
            )
            
            # Store the latest filtered value for waveform visualization
            self.filtered_signal_value = filtered_signal[-1]
            self.waveform_buffer.append(self.filtered_signal_value)
            
            # Step 4: FFT - Frequency domain analysis
            # Use real FFT since signal is real-valued
            fft_data = np.fft.rfft(filtered_signal)
            fft_magnitude = np.abs(fft_data)
            
            # Frequency bins corresponding to FFT output
            freqs = np.fft.rfftfreq(len(filtered_signal), d=1.0/config.FPS)
            
            # Step 5: Find dominant frequency within valid heart rate range
            # Mask frequencies outside our bandpass range
            valid_idx = np.where((freqs >= config.LOW_CUT) & (freqs <= config.HIGH_CUT))[0]
            
            if len(valid_idx) == 0:
                return self.current_bpm
            
            # Find peak frequency in valid range
            valid_fft_magnitude = fft_magnitude[valid_idx]
            valid_freqs = freqs[valid_idx]
            
            peak_idx = np.argmax(valid_fft_magnitude)
            dominant_freq = valid_freqs[peak_idx]
            
            # Convert frequency (Hz) to BPM (beats per minute)
            bpm = dominant_freq * 60.0
            
            # Sanity check: ensure BPM is within physiological range
            if config.LOW_CUT * 60 <= bpm <= config.HIGH_CUT * 60:
                self.current_bpm = bpm
                return round(bpm, 1)
            else:
                return self.current_bpm
                
        except Exception as e:
            print(f"Error calculating BPM: {e}")
            return self.current_bpm
    
    def reset(self) -> None:
        """
        Reset the buffer and BPM estimate.
        Useful when restarting monitoring or switching subjects.
        """
        self.buffer.clear()
        self.current_bpm = None
    
    def get_buffer_size(self) -> int:
        """
        Get the current number of samples in the buffer.
        
        Returns:
            Number of samples currently stored
        """
        return len(self.buffer)
    
    def is_buffer_ready(self) -> bool:
        """
        Check if buffer has enough samples for BPM calculation.
        
        Returns:
            True if buffer is full, False otherwise
        """
        return len(self.buffer) == config.BUFFER_SIZE
    
    def get_waveform_data(self) -> List[float]:
        """
        Get the current waveform data for visualization.
        
        Returns:
            List of filtered signal values for plotting
        """
        return list(self.waveform_buffer)
    
    def get_buffer(self) -> np.ndarray:
        """
        Get the raw signal buffer for analysis or visualization.
        
        Returns:
            Numpy array of raw green channel values from the buffer
        """
        if len(self.buffer) == 0:
            return np.array([], dtype=np.float64)
        
        return np.array(self.buffer, dtype=np.float64)
    
    def get_filtered_signal(self) -> np.ndarray:
        """
        Apply full signal processing pipeline to current buffer and return the filtered signal.
        This is the clean, wave-like signal suitable for visualization.
        
        Pipeline:
        1. Detrending - Remove slow baseline drift
        2. Normalization - Standardize signal amplitude
        3. Bandpass Filtering - Remove noise outside heart rate range
        
        Returns:
            Numpy array of filtered signal values (same length as buffer)
            Returns array of zeros if buffer isn't full yet
        """
        # If buffer isn't full, return zeros
        if len(self.buffer) < config.BUFFER_SIZE:
            return np.zeros(config.BUFFER_SIZE, dtype=np.float64)
        
        try:
            # Convert buffer to numpy array
            raw_signal = np.array(self.buffer, dtype=np.float64)
            
            # Step 1: Detrending - Remove polynomial trend (linear by default)
            detrended_signal = signal.detrend(raw_signal, type='linear')
            
            # Step 2: Normalization - Zero mean, unit variance
            mean_val = np.mean(detrended_signal)
            std_val = np.std(detrended_signal)
            
            # Handle edge case: constant signal (std = 0)
            if std_val < 1e-10:
                return np.zeros(len(raw_signal), dtype=np.float64)
            
            normalized_signal = (detrended_signal - mean_val) / std_val
            
            # Step 3: Bandpass Filtering - Keep only heart rate frequencies
            # Using filtfilt for zero-phase filtering (no time delay)
            filtered_signal = signal.filtfilt(
                self.filter_b,
                self.filter_a,
                normalized_signal
            )
            
            return filtered_signal
            
        except Exception as e:
            print(f"Error generating filtered signal: {e}")
            return np.zeros(config.BUFFER_SIZE, dtype=np.float64)
