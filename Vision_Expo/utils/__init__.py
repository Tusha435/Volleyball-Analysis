"""
Utility functions and helper classes for tracking system.

This package provides essential utilities supporting the volleyball tracking pipeline:
    - ErrorLogger: JSON-based error tracking and reporting
    - ScaleAdapter: Resolution-independent parameter scaling
    - Geometric functions: centroid, IoU, box area calculations
    - CameraMotionCompensator: Homography-based motion estimation
    - Net trackers: Optical flow and homography-based net tracking

These utilities enable the tracker to work across different video resolutions,
compensate for camera movement, and maintain detailed logs for debugging.
"""

from .logger import ErrorLogger
from .scale_adapter import ScaleAdapter
from .geometry import centroid, iou, box_area
from .camera_motion import CameraMotionCompensator
from .net_tracker import OpticalFlowNetTracker, NetTracker

__all__ = [
    'ErrorLogger', 'ScaleAdapter', 'centroid', 'iou', 'box_area',
    'CameraMotionCompensator', 'OpticalFlowNetTracker', 'NetTracker'
]
