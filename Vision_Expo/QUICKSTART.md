# Quick Start Guide

Get started with Vision_Expo in 5 minutes!

## 🚀 Installation

```bash
# Install dependencies
pip install opencv-python numpy scipy filterpy ultralytics torch

# Optional: Install SAM2 for segmentation
git clone https://github.com/facebookresearch/segment-anything-2
cd segment-anything-2
pip install -e .
```

## ⚙️ Configuration

Edit [config.py](config.py):

```python
CONFIG = {
    'video_path': r"path/to/your/video.mp4",     # ← Change this
    'output_path': r"path/to/output.mp4",        # ← Change this
    'device': "cuda",  # or "cpu"
}
```

## ▶️ Run

```bash
python main.py
```

## 🎮 Setup Steps

1. **Net Line**: Click 2 points on the net → Press ENTER
2. **Referee Zones**: Click 2 referee positions → Press ENTER
3. **Processing**: Automatic!

## ⌨️ Controls

- `q` - Quit
- `s` - Save screenshot

## 📊 Output

- Video: Saved to `output_path`
- Logs: `output/error_report.json`

## 🎯 Expected Results

- **Players**: Magenta (Team A) / Green (Team B) boxes
- **Referees**: Yellow boxes labeled `REF_1`, `REF_2`
- **Ball**: Green circle with trajectory trail
- **Net**: Cyan line
- **Info Panel**: Count summary (top-right)

## 🐛 Troubleshooting

**Issue**: Video won't open
```python
# Try changing codec in config.py
CONFIG['output_path'] = r"path/to/output.avi"  # Use .avi instead
```

**Issue**: Poor tracking
```python
# Adjust detection confidence
CONFIG['conf_person'] = 0.25  # Lower = more detections
CONFIG['conf_ball'] = 0.05    # Lower for hard-to-see balls
```

**Issue**: SAM2 errors
```python
# Disable segmentation
CONFIG['use_sam2'] = False
```

## 📚 Next Steps

- Full documentation: [README.md](README.md)
- Technical details: [docs/Formula_understanding.md](docs/Formula_understanding.md)
- Advanced tuning: See [config.py](config.py) parameters

## 💡 Tips

1. **Good Lighting**: Ensures better detection
2. **Stable Camera**: Improves homography estimation
3. **Clear Net View**: Makes team assignment more accurate
4. **Referee Zones**: Place at actual referee standing positions

---

**Need Help?** Check the full [README.md](README.md) or open an issue!
