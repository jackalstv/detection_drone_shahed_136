# 🎯 Drone Detective - Shahed-136 Detection System

> **Low-cost AI-powered drone detection system achieving 99.5% accuracy**  
> Developed during CyberDefense Hackathon 2024 (42 hours)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![mAP50](https://img.shields.io/badge/mAP50-99.5%25-success.svg)](.)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance](#-performance)
- [Deployment](#-deployment)
- [Team](#-team)
- [License](#-license)

---

## 🚨 Problem Statement

Since 2022, Ukraine has faced massive attacks from Iranian Shahed-136 kamikaze drones. These drones pose critical challenges:

- **Difficult to detect:** Low altitude flight (50-300m), moderate speed (185 km/h)
- **Asymmetric cost:** One Shahed costs ~€20,000, while intercepting with Patriot missiles costs €4 million
- **Defense saturation:** Launched in swarms of dozens simultaneously
- **Coverage gaps:** Military radars don't cover all civilian infrastructure

**Our mission:** Create an accessible, low-cost detection solution deployable on existing camera infrastructure.

---

## 💡 Our Solution

**Drone Detective** transforms any surveillance camera into an intelligent drone detector using artificial intelligence. No additional sensors required.

### What it does:
1. **Detects** Shahed-136 drones automatically with 99.5% precision
2. **Estimates distance** in real-time using computer vision (no LiDAR/radar needed)
3. **Tracks** drones continuously, even through temporary occlusions
4. **Alerts** defense systems or populations for early warning

### Key advantage:
Deployable in hours on existing camera networks (urban surveillance, security cameras) for **under €1,000 per point** vs €500,000+ for military radars.

---

## ⚡ Key Features

### 🎯 Detection
- **YOLOv11 custom model** trained on 930 military drone images
- **99.5% mAP50** accuracy (near-perfect detection)
- **99.9% recall** (catches virtually all drones)
- **Anti-false-positive filters** (geometric validation, confidence thresholds)

### 📏 Distance Estimation
- **3D geometric calculation** without additional sensors
- **Multi-dimension weighted approach** (adapts to viewing angle)
- **Temporal smoothing** (20-frame median filter)
- **Outlier rejection** (< 15% coefficient of variation)
- **Real-time confidence intervals** displayed

### 🎬 Smart Tracking
- **Temporal memory** (maintains track through 5-frame gaps)
- **IoU matching** (30% threshold for robust tracking)
- **Unique counting** (no duplicate detections)
- **Occlusion handling** (continues tracking through temporary obstructions)

### 💻 User Interface
- **Real-time HUD** with FPS, frame count, detection statistics
- **Visual feedback** (bounding boxes, keypoints, tracking status)
- **Debug mode** with detailed metrics
- **Keyboard controls** (pause, reset, quit)

---

## 🏗️ Technical Architecture

### Stack
```
Language:     Python 3.12
AI Framework: Ultralytics YOLOv11 (PyTorch)
Vision:       OpenCV 4.x
Computing:    NumPy
```

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT (Camera Feed)                    │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │  YOLOv11 Detection   │
         │  (99.5% mAP50)       │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │  Geometric Filters   │
         │  (Anti-FP)           │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │  Smart Tracker       │
         │  (IoU + Memory)      │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │  3D Distance Calc    │
         │  (Multi-dimension)   │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │  Temporal Smoothing  │
         │  (Median Filter)     │
         └───────────┬──────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              OUTPUT (Alerts + Visualization)             │
└─────────────────────────────────────────────────────────┘
```

### Distance Estimation Method

The system calculates distance using **triangulation without additional sensors**:

```
Known: Shahed-136 dimensions
  - Length: 3.5 m
  - Wingspan: 2.5 m

Measured: Apparent size in image (pixels)

Formula: Distance = (Real_size × Focal_length) / Apparent_size

Improvements:
  ✓ Auto-detects viewing angle (lateral/frontal/oblique)
  ✓ Weighted calculation using 3 dimensions
  ✓ Temporal smoothing (median over 20 frames)
  ✓ Outlier rejection (>2.5× median deviation)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU (optional, but recommended for real-time performance)
- Webcam or video file for testing

### Quick Setup

```bash
# Clone repository
git clone https://github.com/jackalstv/detection_drone_shahed_136.git
cd detection_drone_shahed_136

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install ultralytics opencv-python numpy

# Download dataset (optional, for training)
# Visit: https://universe.roboflow.com/yana-leshch-prjr4/military-units-24v07v2023
# Extract to: Military-Units-24v07v2023.v1i.yolov11/
```

---

## 📖 Usage

### 1. Training (Optional - Model Included)

```bash
# Train custom YOLOv11 model
python trainmodel.py

# Output: shahed136_detector.pt (5.2 MB)
# Training time: ~2h on GPU, ~8h on CPU
```

### 2. Detection on Video

```bash
# Run on video file
python main.py --video your_video.mp4

# Run on webcam
python main.py --video 0

# With custom confidence threshold
python main.py --video video.mp4 --conf 0.6

# Debug mode (detailed metrics)
python main.py --video video.mp4 --debug
```

### 3. Calibration (Optional)

If distances seem inaccurate, calibrate using camera FOV and manual factor:

```bash
# Adjust Field of View (default: 60°)
python main.py --fov 50  # Narrower FOV → larger distances
python main.py --fov 70  # Wider FOV → smaller distances

# Manual calibration factor
python main.py --calib 2.0  # Multiply distances by 2
python main.py --calib 0.5  # Divide distances by 2

# Combined calibration
python main.py --fov 55 --calib 1.5 --debug
```

### 4. Keyboard Controls

While running:
- **Q**: Quit
- **SPACE**: Pause/Resume
- **R**: Reset statistics

---

## 📊 Performance

### Detection Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **mAP50** | 99.5% | Near-perfect |
| **mAP50-95** | 98.0% | Excellent |
| **Precision** | 99.9% | Minimal false positives |
| **Recall** | 99.9% | Catches virtually all drones |

### Distance Estimation

| Metric | Value |
|--------|-------|
| **Method** | Multi-dimension geometric |
| **Stability** | < 15% coefficient of variation |
| **Confidence** | 5-95% percentile displayed |
| **Smoothing** | 20-frame median filter |

### Runtime Performance

| Hardware | FPS | Notes |
|----------|-----|-------|
| Laptop CPU (Ryzen 7 5825U) | 5 FPS | Debug mode |
| Desktop GPU (RTX 3060) | 30+ FPS | Real-time capable |
| Raspberry Pi 5 | 15+ FPS | Estimated (not tested) |

---

## 🌍 Deployment

### Cost Analysis

| Solution | Cost per Point | Notes |
|----------|----------------|-------|
| **Drone Detective** | €200-1,000 | Using existing cameras |
| Military radar | €500,000-2M | Mobile systems |
| Commercial anti-drone | €50,000-200,000 | Dedicated hardware |

**Cost advantage: 50-2000× cheaper**

### Deployment Scenarios

#### Scenario A: Existing Infrastructure
```
Reuse existing cameras:
  - Municipal surveillance (intersections, squares)
  - Critical infrastructure (power plants, bridges)
  - Private security cameras (with consent)

Requirements:
  - Mini-PC (Intel NUC, Raspberry Pi 5): €200-500
  - Software installation: 1 hour
  
Total: €200-500 per point
```

#### Scenario B: New Installation
```
Complete system:
  - Outdoor IP camera: €150
  - Mini-PC: €300
  - Installation: €200
  
Total: €650 per point
```

### Network Architecture

```
[Local Cameras] → [Edge Processing] → [Central Server]
                   (Mini-PC)              ↓
                                     [Dashboard]
                                     [SMS/App Alerts]
                                     [Defense Integration]
```

**Advantages:**
- ✅ Local processing (low bandwidth)
- ✅ Resilient (continues if network fails)
- ✅ Scalable (add cameras without server overload)

### Production Capacity

| Phase | Target | Coverage |
|-------|--------|----------|
| Pilot | 100 units/month | Medium city |
| Scale | 1,000 units/month | Region |
| Mass | 10,000 units/month | Country |

---

## 👥 Team

**Drone Detective** - 4 members  
Developed during CyberDefense Hackathon 2024 (42 hours)

### Timeline
- **Friday 6pm-midnight:** Dataset research and validation
- **Saturday 9am-midnight:** Model training (99.5% mAP50 achieved!)
- **Saturday midnight-3am:** Distance estimation implementation
- **Sunday 9am-noon:** Tracking system, smoothing, UI finalization

### Challenges Overcome
- ✅ False positives (solved with geometric filters) - 3h
- ✅ Unstable PnP distance calculation (switched to robust geometric method) - 4h
- ✅ Tracking instability (implemented IoU matching + memory) - 2h
- ✅ Variable distances by angle (weighted multi-dimension calculation) - 3h

**Total development obstacles: 13h / 42h (31%)**

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🔗 Links

- **Dataset:** [Military-Units-24v07v2023](https://universe.roboflow.com/yana-leshch-prjr4/military-units-24v07v2023)
- **YOLOv11:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Documentation:** [Full Report](Drone_detective.pdf)

---

## 🎓 Citation

If you use this project in your research or development:

```bibtex
@software{drone_detective_2024,
  title = {Drone Detective: Low-Cost AI Shahed-136 Detection System},
  author = {Drone Detective Team},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jackalstv/detection_drone_shahed_136}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This project was developed for educational and defensive purposes during a hackathon. It is intended to protect civilian infrastructure and does not include any offensive capabilities.

---

## 📧 Contact

For questions, collaboration, or deployment inquiries:
- **GitHub Issues:** [Open an issue](https://github.com/jackalstv/detection_drone_shahed_136/issues)
- **Team:** Drone Detective

---

<div align="center">

**Built with ❤️ during CyberDefense Hackathon 2024**

*Technology serving to protect civilians*

</div>
