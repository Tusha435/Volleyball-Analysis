# Vision_Expo - Volleyball Tracking System

A comprehensive, production-ready volleyball tracking system using computer vision, deep learning, and advanced tracking algorithms.

## 🎯 Overview

Vision_Expo is a modular volleyball tracking application that provides:
- **Player tracking** with team assignment (Team A vs Team B)
- **Referee detection** using position-locked classification
- **Ball tracking** with trajectory visualization
- **SAM2 segmentation** for precise player masking
- **Camera motion compensation** using homography estimation
- **Net line tracking** via optical flow

## 📁 Project Structure

```
Vision_Expo/
├── main.py                          # Entry point
├── config.py                        # Configuration settings
├── volleyball_tracker.py            # Main tracker application
│
├── core/                            # Core tracking modules
│   ├── __init__.py
│   ├── detector.py                  # YOLO-based object detection
│   ├── tracker.py                   # Multi-object tracker with Hungarian algorithm
│   ├── track.py                     # Track state management with Kalman filtering
│   └── segmentor.py                 # SAM2 segmentation
│
├── classifiers/                     # Classification modules
│   ├── __init__.py
│   ├── referee_classifier.py       # Position-locked referee detection
│   └── team_assigner.py            # Team assignment based on net position
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── logger.py                    # Error logging
│   ├── scale_adapter.py             # Resolution scaling
│   ├── geometry.py                  # Geometric calculations (IoU, centroid, etc.)
│   ├── camera_motion.py             # Homography-based camera motion estimation
│   └── net_tracker.py               # Net line tracking (optical flow + homography)
│
└── docs/                            # Documentation
    ├── README.md                    # This file
    └── Formula_understanding.md     # Technical formulas and algorithms
```

## 🚀 Features

### 1. **Multi-Object Tracking**
- Uses **Hungarian algorithm** (linear sum assignment) for optimal track-detection matching
- **Kalman filtering** for smooth position prediction
- Separate trackers for players, referees, and ball

### 2. **Position-Locked Referee Detection**
- Referees are identified by their position on court (manually selected at startup)
- Camera motion compensation keeps referee zones accurate
- Hysteresis mechanism prevents false demotions (10-frame strike tolerance)

### 3. **Team Assignment**
- Automatic team classification based on court side (relative to net)
- Voting system for stable team assignment
- Visual color coding: Team A (magenta), Team B (green)

### 4. **Ball Tracking**
- Single ball enforcement (keeps only best candidate)
- Trajectory history with fade effect
- High sensitivity detection for small objects

### 5. **Camera Motion Compensation**
- SIFT/ORB feature matching for homography estimation
- Net position updates via optical flow
- Stabilizes tracking during camera pans/zooms

### 6. **SAM2 Segmentation** (Optional)
- Precise player masks for better visualization
- Box-prompted segmentation
- Graceful fallback if unavailable

## 🛠️ Installation

### Prerequisites
```bash
pip install opencv-python numpy scipy filterpy ultralytics torch
```

### Optional (for SAM2 segmentation)
```bash
# Install SAM2
git clone https://github.com/facebookresearch/segment-anything-2
cd segment-anything-2
pip install -e .

# Download SAM2 checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_base_plus.pt
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    'video_path': r"path/to/your/video.mp4",
    'model_path': "rtdetr-x.pt",           # YOLO model
    'device': "cuda",                       # or "cpu"
    'output_path': r"path/to/output.mp4",

    # Detection thresholds
    'conf_person': 0.30,
    'conf_ball': 0.10,

    # Tracking parameters
    'max_age_player': 30,
    'max_age_ref': 120,
    'max_age_ball': 10,
    'iou_player': 0.30,
    'iou_ref': 0.25,
    'iou_ball': 0.15,

    # Camera motion
    'use_homography': True,
    'feature_detection': 'SIFT',           # or 'ORB'

    # SAM2 segmentation
    'use_sam2': True,
    'sam2_checkpoint': r"path/to/sam2.pt",
    'sam2_config': r"path/to/sam2_config.yaml",

    # Error logging
    'log_errors': True,
    'error_log_path': r"path/to/error_report.json",
}
```

## 🎮 Usage

### Basic Usage
```bash
python main.py
```

### Setup Process
1. **Net Selection**: Click 2 points to define the net line (press ENTER)
2. **Referee Zones**: Click EXACTLY 2 positions for referee detection zones (press ENTER)
3. Processing begins automatically

### Keyboard Controls
- `q` - Quit processing
- `s` - Save screenshot

## 📊 Output

### Video Output
- Annotated video saved to `output_path`
- Bounding boxes color-coded by role/team
- Net line visualization (cyan)
- Ball trajectory trail (green)
- Info panel showing counts

### Error Log (JSON)
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "total_errors": 5,
  "errors_by_type": {
    "detection_failure": 3,
    "tracking_loss": 2
  },
  "errors": [...]
}
```

## 🧩 Module Details

### Core Modules

#### `detector.py`
- YOLO-based detection (RT-DETR or YOLOv8)
- Custom NMS (Non-Maximum Suppression)
- Validation filters for size/aspect ratio

#### `tracker.py`
- Hungarian algorithm for assignment
- Multi-metric cost matrix (IoU + distance + size)
- Motion consistency checks

#### `track.py`
- Kalman filter (6D state: x, y, vx, vy, ax, ay)
- Stability metrics (position/size variance)
- History management

### Classifier Modules

#### `referee_classifier.py`
- Position-based detection zones
- Camera motion compensation
- Candidate scoring system

#### `team_assigner.py`
- Cross-product based side detection
- Voting mechanism for stability

### Utility Modules

#### `camera_motion.py`
- Feature matching (SIFT/ORB)
- RANSAC homography estimation
- Confidence scoring

#### `net_tracker.py`
- Optical flow tracking (Lucas-Kanade)
- Homography-based fallback
- Stability detection

## 🔬 Technical Highlights

### Tracking Algorithm
- **Cost Matrix**: Combines IoU (50%), distance (30%), size similarity (20%)
- **Kalman Filter**: Constant acceleration model for smooth predictions
- **Confirmation**: Requires 3 hits in 3 frames before track is confirmed

### Referee Detection
- **Position Locking**: Tracks must be within radius of manual zones
- **Scoring**: Distance (10pts) + stability (3pts) + history (5pts) + age (2pts)
- **Hysteresis**: 10 consecutive strikes required before demotion

### Ball Tracking
- **Single Ball Enforcement**: Keeps highest-scored ball (hits - misses + trajectory)
- **Trajectory**: 100-frame history with visual fade effect

## 🐛 Troubleshooting

### Common Issues

**No video output**
- Check codec support (try XVID if mp4v fails)
- Verify write permissions

**Poor tracking**
- Adjust confidence thresholds in config
- Check lighting conditions
- Verify YOLO model quality

**SAM2 not working**
- Ensure correct installation path
- Check CUDA availability
- Set `use_sam2: False` to disable

**Camera motion issues**
- Try 'ORB' instead of 'SIFT'
- Disable homography: `use_homography: False`

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Key areas:
- Multi-camera support
- 3D ball trajectory estimation
- Action recognition
- Performance optimization

## 📧 Contact

For questions or support, please open an issue on the repository.

---

**Version**: 1.0.0
**Author**: Tushar Sinha
**Last Updated**: 2025
