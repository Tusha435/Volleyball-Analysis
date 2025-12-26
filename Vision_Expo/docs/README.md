# Vision_Expo Documentation

Welcome to the Vision_Expo documentation directory!

## Available Documentation

### 1. [Main README](../README.md)
Complete project overview, installation guide, usage instructions, and feature descriptions.

**Contents**:
- Project overview and features
- Installation instructions
- Configuration guide
- Usage and keyboard controls
- Module descriptions
- Troubleshooting guide

### 2. [Formula Understanding](Formula_understanding.md)
Technical deep dive into all mathematical formulas and algorithms used in the system.

**Contents**:
- Geometric calculations (IoU, centroid, distance)
- Kalman filtering equations
- Hungarian algorithm and cost matrix
- Camera motion compensation (homography)
- Referee classification scoring
- Team assignment (cross product method)
- Optical flow (Lucas-Kanade)
- Non-Maximum Suppression
- Stability metrics
- Scale adaptation formulas

## Quick Links

### For Users
- Start with the [Main README](../README.md)
- See configuration options in [config.py](../config.py)
- Run the application: `python main.py`

### For Developers
- Understand algorithms in [Formula Understanding](Formula_understanding.md)
- Explore module architecture in [Main README - Module Details](../README.md#-module-details)
- Check code organization in respective module directories:
  - [core/](../core/) - Detection and tracking
  - [classifiers/](../classifiers/) - Referee and team classification
  - [utils/](../utils/) - Helper utilities

### For Researchers
- Algorithm details: [Formula Understanding](Formula_understanding.md)
- Performance tuning: See parameter tables in both documents
- Extension points: Check module interfaces in source code

## Documentation Maintenance

When updating the codebase:
1. Update relevant formulas in `Formula_understanding.md`
2. Update feature descriptions in main `README.md`
3. Update parameter tables if defaults change
4. Add new modules to architecture diagrams

---

**Last Updated**: 2024
**Documentation Version**: 1.0.0
