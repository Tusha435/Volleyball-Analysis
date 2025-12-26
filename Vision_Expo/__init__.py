"""
Vision_Expo - Professional Volleyball Tracking System.

A modular, production-ready computer vision system for tracking players, referees,
and ball in volleyball game footage. Features include multi-object tracking,
team assignment, referee identification, camera motion compensation, and optional
SAM2 segmentation.

Key Components:
    - YOLO-based object detection with custom NMS
    - Hungarian algorithm for optimal track-detection matching
    - Kalman filtering for smooth position prediction
    - Position-locked referee classification
    - Geometric team assignment based on court position
    - Homography-based camera motion compensation
    - Optical flow net tracking

For usage information, see README.md in the project root.
"""

__version__ = "1.0.0"
__author__ = "Vision_Expo Team"
__all__ = ['VolleyballTrackerPro']

from .volleyball_tracker import VolleyballTrackerPro
