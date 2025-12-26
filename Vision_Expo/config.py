"""
Global configuration for the volleyball tracking system.

This module contains all runtime parameters including paths, model settings,
detection thresholds, and tracking parameters. Modify these values to tune
the tracker's behavior for different videos or requirements.
"""

CONFIG = {
    'video_path': r"D:\volleyball_vision\data\sample_videos\input_video_2.mp4",
    'model_path': "rtdetr-x.pt",
    'device': "cuda",
    'output_path': r"D:\volleyball_vision\output\finale_no_1.mp4",

    'reference_width': 1920,
    'reference_height': 1080,

    'sam2_checkpoint': r"D:\volleyball_vision\sam2.1_hiera_base_plus.pt",
    'sam2_config': r"D:\volleyball_vision\segment_anything_2\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml",
    'use_sam2': True,

    'conf_person': 0.30,
    'conf_ball': 0.10,

    'max_age_player': 30,
    'max_age_ref': 120,
    'max_age_ball': 10,

    'iou_player': 0.30,
    'iou_ref': 0.25,
    'iou_ball': 0.15,

    'use_homography': True,
    'feature_detection': 'SIFT',

    'log_errors': True,
    'error_log_path': r"D:\volleyball_vision\output\error_report.json",
}
