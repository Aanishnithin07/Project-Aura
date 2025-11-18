# 🩺 AuraScan: Remote Vitals Monitor

**Contactless Heart Rate Monitoring via Remote Photoplethysmography (rPPG)**

AuraScan is a web-based telehealth application that detects your heart rate through your webcam by analyzing subtle color changes in facial skin caused by blood flow.

![AuraScan Banner](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **Real-time Heart Rate Detection**: Contactless BPM monitoring through webcam
- **Professional HUD Interface**: ECG-style waveform visualization
- **Web-Based**: No installation required - runs in your browser
- **Privacy-First**: All processing happens locally in your browser
- **Medical-Grade Visualization**: Real-time pulse waveform display
- **Responsive Design**: Works on desktop and mobile devices

## 🔬 How It Works

AuraScan uses **Remote Photoplethysmography (rPPG)** technology:

1. **Blood Flow Detection**: Each heartbeat pumps blood through facial capillaries, causing tiny color changes
2. **Forehead ROI**: Analyzes a region of interest on your forehead (high capillary density, minimal movement)
3. **Green Channel Extraction**: Blood absorbs green light strongly - we track green channel intensity
4. **Digital Signal Processing**: 
   - Detrending removes baseline shifts
   - Bandpass filter (0.7-4.0 Hz) isolates heart rate frequencies
   - FFT analysis finds the dominant frequency
   - Frequency × 60 = BPM
5. **Real-time Display**: ECG-style waveform shows your pulse in real-time

**Typical Accuracy**: ±2-5 BPM compared to medical pulse oximeters under optimal conditions.

## 🚀 Quick Start

### Desktop Application

```bash
# Clone the repository
git clone https://github.com/Aanishnithin07/Project-Aura.git
cd Project-Aura

# Create virtual environment (Python 3.12 required)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run desktop app
python main.py
```

### Web Application (Streamlit)

```bash
# Install web dependencies
pip install -r requirements_web.txt

# Run Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Deploy**:
   - Click "New app"
   - Select your repository: `YourUsername/Project-Aura`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Share** your app URL with others!

## 📋 Requirements

### Desktop App
- Python 3.12
- OpenCV 4.12+
- MediaPipe 0.10.14
- NumPy, SciPy

### Web App (Additional)
- Streamlit 1.40+
- streamlit-webrtc 0.47+
- PyAV 14.0+

## 🎯 Best Practices for Accurate Readings

✅ **Good Lighting**: Use natural or bright indoor lighting  
✅ **Stay Still**: Minimize head movement during measurement  
✅ **Face Camera**: Keep your face centered and fully visible  
✅ **Forehead Visible**: Ensure forehead is not covered by hair/accessories  
✅ **Wait**: Allow 5-10 seconds for signal acquisition  
✅ **Stable Camera**: Use a tripod or stable surface  

## 📊 Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Webcam Input (30 FPS)                │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Face Detector (MediaPipe Face Mesh)                    │
│  • 468 facial landmarks                                 │
│  • Forehead ROI extraction (36 landmarks)               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Signal Processor (rPPG Pipeline)                       │
│  • Green channel extraction                             │
│  • Detrending (polynomial fit)                          │
│  • Normalization                                        │
│  • Butterworth bandpass filter (0.7-4.0 Hz)            │
│  • FFT frequency analysis                               │
│  • BPM calculation                                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  UI Renderer (Professional HUD)                         │
│  • ECG-style waveform                                   │
│  • Real-time BPM display                                │
│  • Status indicators                                    │
│  • Progress bars                                        │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
AuraScan/
├── app.py                      # Streamlit web application
├── main.py                     # Desktop application
├── face_detector.py            # MediaPipe face detection
├── signal_processor.py         # rPPG signal processing
├── ui_renderer.py              # HUD visualization
├── config.py                   # Configuration constants
├── requirements.txt            # Desktop dependencies
├── requirements_web.txt        # Web app dependencies
├── packages.txt                # System dependencies (Streamlit Cloud)
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
└── README_WEB.md              # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
BUFFER_SIZE = 150        # Signal buffer size (frames)
FPS = 30                 # Camera frame rate
FILTER_LOW_HZ = 0.7      # Min heart rate: 42 BPM
FILTER_HIGH_HZ = 4.0     # Max heart rate: 240 BPM
MIN_BPM = 40             # Minimum valid BPM
MAX_BPM = 200            # Maximum valid BPM
```

## 🐛 Troubleshooting

### Camera not working in web app
- Grant camera permissions in your browser
- Use HTTPS (required for camera access on non-localhost)
- Try a different browser (Chrome/Edge recommended)

### Low accuracy
- Improve lighting conditions
- Reduce head movement
- Ensure forehead is clearly visible
- Wait longer for signal acquisition

### Dependencies issues
```bash
# For MediaPipe compatibility, use Python 3.12
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_web.txt
```

## ⚠️ Disclaimer

**For Research & Educational Use Only**

This is **NOT a medical device**. Do not use for clinical diagnosis or medical decision-making. Always consult healthcare professionals for medical concerns.

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for face detection
- **Remote Photoplethysmography** research community
- **Streamlit** for the web framework
- **OpenCV** for image processing

## 📞 Contact

**Project Aura** - Telehealth Innovation

- GitHub: [@Aanishnithin07](https://github.com/Aanishnithin07)
- Repository: [Project-Aura](https://github.com/Aanishnithin07/Project-Aura)

## 🚀 Future Roadmap

- [ ] Respiratory rate detection
- [ ] Blood oxygen estimation (SpO2)
- [ ] Stress level analysis
- [ ] Multi-user support
- [ ] Mobile app (iOS/Android)
- [ ] Data export & analytics dashboard
- [ ] Integration with health platforms (Apple Health, Google Fit)

---

**Made with ❤️ for the future of telehealth**
