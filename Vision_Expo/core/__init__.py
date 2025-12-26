"""
Core object detection and tracking components.

This package contains the fundamental building blocks for multi-object tracking:
    - EnhancedDetector: YOLO-based detection with custom NMS and validation
    - Tracker: Hungarian algorithm-based multi-object tracker
    - Track: Individual track state with Kalman filtering
    - SAM2Segmentor: Optional segmentation for precise object masks

The tracking system uses a predict-update cycle with sophisticated cost matrix
computation for optimal assignment between detections and existing tracks.
"""

from .detector import EnhancedDetector
from .tracker import Tracker
from .track import Track
from .segmentor import SAM2Segmentor, SAM2_AVAILABLE

__all__ = ['EnhancedDetector', 'Tracker', 'Track', 'SAM2Segmentor', 'SAM2_AVAILABLE']
