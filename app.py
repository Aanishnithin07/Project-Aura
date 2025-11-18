"""
AuraScan Streamlit Web Application
Remote Photoplethysmography (rPPG) Heart Rate Monitor

This web app uses your webcam to detect heart rate through facial color changes.
Based on remote photoplethysmography technology.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from collections import deque
import threading

# Import custom modules
from face_detector import FaceDetector
from signal_processor import SignalProcessor
from ui_renderer import UIRenderer
import config


# Configure Streamlit page
st.set_page_config(
    page_title="AuraScan: Remote Vitals Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff64;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #00c8ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #00ff64;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #00ff64;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #00ff64;
    }
    </style>
""", unsafe_allow_html=True)


# Main header
st.markdown('<p class="main-header">🩺 AuraScan: Remote Vitals Monitor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Contactless Heart Rate Monitoring via Remote Photoplethysmography</p>',
    unsafe_allow_html=True
)


# Information banner
st.info(
    "📹 **How it works:** AuraScan analyzes subtle color changes in your facial skin caused by blood flow. "
    "Position your face in the camera frame and wait 5-10 seconds for signal acquisition."
)


class VideoTransformer(VideoProcessorBase):
    """
    Real-time video processor for heart rate detection.
    Processes each video frame to detect face, extract signal, and calculate BPM.
    """

    def __init__(self):
        # Initialize detection and processing modules
        self.face_detector = FaceDetector()
        self.signal_processor = SignalProcessor()
        self.ui_renderer = UIRenderer()
        
        # Thread-safe lock for shared state
        self.lock = threading.Lock()
        
        # Tracking variables
        self.current_bpm = 0
        self.face_detected = False
        self.buffer_fill_percentage = 0
        self.frame_count = 0
        
        # Streamlit UI elements (will be updated from main thread)
        self.status_placeholder = None
        self.metrics_placeholder = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process each incoming video frame.
        
        Args:
            frame: Input video frame from webcam
            
        Returns:
            Processed video frame with overlays
        """
        # Convert AVFrame to NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        # Create a copy for processing
        display_frame = img.copy()
        
        # Increment frame counter
        self.frame_count += 1
        
        # Detect face and get landmarks
        landmarks = self.face_detector.get_landmarks(img)
        
        with self.lock:
            if landmarks is not None:
                self.face_detected = True
                
                # Extract Region of Interest (forehead area)
                roi_pixels = self.face_detector.extract_roi_pixels(img, landmarks)
                
                if roi_pixels is not None and len(roi_pixels) > 0:
                    # Calculate average green channel value (blood volume proxy)
                    green_avg = np.mean(roi_pixels[:, 1])  # Index 1 is Green channel in BGR
                    
                    # Add to signal processor buffer
                    self.signal_processor.add_value(green_avg)
                    
                    # Calculate buffer fill percentage
                    buffer_size = len(self.signal_processor.buffer)
                    self.buffer_fill_percentage = (buffer_size / config.BUFFER_SIZE) * 100
                    
                    # Draw ROI rectangle on frame
                    roi_rect = self.face_detector.get_roi_rect(landmarks)
                    if roi_rect is not None:
                        x, y, w, h = roi_rect
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
                        cv2.putText(
                            display_frame,
                            "ROI",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 100),
                            2
                        )
                    
                    # Calculate BPM if buffer is sufficiently filled
                    if buffer_size >= config.BUFFER_SIZE:
                        bpm = self.signal_processor.calculate_bpm()
                        if bpm is not None:
                            self.current_bpm = bpm
                    
                    # Get filtered signal for waveform visualization
                    filtered_signal = self.signal_processor.get_filtered_signal()
                    
                    # Draw waveform graph
                    if filtered_signal is not None and len(filtered_signal) > 0:
                        graph_x = 20
                        graph_y = 20
                        self.ui_renderer.draw_graph(
                            display_frame,
                            filtered_signal,
                            (graph_x, graph_y)
                        )
                    
                    # Draw info panel with BPM
                    self.ui_renderer.draw_info_panel(
                        display_frame,
                        self.current_bpm,
                        self.face_detected
                    )
                    
                    # Draw progress bar for buffer fill
                    if self.buffer_fill_percentage < 100:
                        progress_x = display_frame.shape[1] - 220
                        progress_y = display_frame.shape[0] - 60
                        self.ui_renderer.draw_progress_bar(
                            display_frame,
                            self.buffer_fill_percentage / 100,
                            (progress_x, progress_y),
                            200,
                            20
                        )
            else:
                # No face detected
                self.face_detected = False
                self.buffer_fill_percentage = 0
                
                # Draw "searching" status
                self.ui_renderer.draw_info_panel(
                    display_frame,
                    0,
                    False
                )
        
        # Convert back to AVFrame
        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📹 Live Camera Feed")
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="aura_scan",
        video_processor_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Status display
    if webrtc_ctx.state.playing:
        status_container = st.container()
        with status_container:
            st.success("✅ Camera Active - Analyzing facial blood flow patterns")
            
            # Progress indicator
            progress_placeholder = st.empty()
            
            if webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor
                if hasattr(processor, 'buffer_fill_percentage'):
                    if processor.buffer_fill_percentage < 100:
                        progress_placeholder.progress(
                            processor.buffer_fill_percentage / 100,
                            text=f"Acquiring Signal: {processor.buffer_fill_percentage:.0f}%"
                        )
                    else:
                        progress_placeholder.success("Signal Acquired - Real-time monitoring active")

with col2:
    st.markdown("### 📊 Live Metrics")
    
    # Metrics display
    metrics_placeholder = st.empty()
    
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor
        
        with metrics_placeholder.container():
            # BPM metric
            if hasattr(processor, 'current_bpm') and processor.current_bpm > 0:
                st.metric(
                    label="Heart Rate",
                    value=f"{processor.current_bpm:.0f} BPM",
                    delta=None
                )
            else:
                st.metric(
                    label="Heart Rate",
                    value="-- BPM",
                    delta="Acquiring..."
                )
            
            # Face detection status
            if hasattr(processor, 'face_detected'):
                if processor.face_detected:
                    st.success("✅ Face Locked")
                else:
                    st.warning("🔍 Searching...")
            
            # Buffer status
            if hasattr(processor, 'buffer_fill_percentage'):
                st.metric(
                    label="Signal Quality",
                    value=f"{processor.buffer_fill_percentage:.0f}%"
                )


# Sidebar with instructions
with st.sidebar:
    st.markdown("## 📋 Instructions")
    
    st.markdown("""
    ### How to Use:
    1. **Click "START"** to activate your webcam
    2. **Position your face** in the camera frame
    3. **Wait 5-10 seconds** for signal acquisition
    4. **View your heart rate** in real-time
    
    ### Best Practices:
    ✅ **Good Lighting**: Use natural or bright indoor lighting  
    ✅ **Stay Still**: Minimize head movement  
    ✅ **Face Camera**: Keep face centered and visible  
    ✅ **Forehead Visible**: Ensure forehead is not covered  
    ✅ **Stable Position**: Use a tripod or stable surface  
    
    ### Technical Details:
    🔬 **Technology**: Remote Photoplethysmography (rPPG)  
    📊 **Accuracy**: ±2-5 BPM (under optimal conditions)  
    🎯 **Detection**: 30 FPS real-time processing  
    💚 **Signal**: Green light absorption in facial capillaries  
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔬 How It Works")
    with st.expander("The Science Behind AuraScan"):
        st.markdown("""
        **Remote Photoplethysmography (rPPG)**
        
        1. **Blood Flow Detection**:
           - Each heartbeat pumps blood through facial capillaries
           - This causes tiny color changes in your skin
           
        2. **Forehead Analysis**:
           - High capillary density
           - Minimal movement interference
           - Consistent skin exposure
           
        3. **Signal Processing**:
           - Extract green channel (blood absorbs green light)
           - Average pixel values from forehead ROI
           - Apply digital filters (0.7-4.0 Hz bandpass)
           - FFT analysis to find dominant frequency
           - Convert frequency to BPM
           
        4. **Accuracy Factors**:
           - Good lighting improves accuracy
           - Stable camera reduces noise
           - 5-10 second buffer ensures reliability
        """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "**For Research & Educational Use Only**\n\n"
        "This is not a medical device. Do not use for clinical diagnosis. "
        "Consult healthcare professionals for medical concerns."
    )
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Configuration")
    st.json({
        "Buffer Size": config.BUFFER_SIZE,
        "FPS": config.FPS,
        "Filter Range": f"{config.FILTER_LOW_HZ}-{config.FILTER_HIGH_HZ} Hz",
        "BPM Range": f"{config.MIN_BPM}-{config.MAX_BPM}",
        "Resolution": f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"
    })


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "🚀 Project Aura | Built with ❤️ using Streamlit, OpenCV & MediaPipe | "
    "<a href='https://github.com/Aanishnithin07/Project-Aura'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)
