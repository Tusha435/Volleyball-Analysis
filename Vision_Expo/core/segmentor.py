"""
SAM2 segmentation module for precise object boundary detection.

This module provides integration with the Segment Anything Model 2 (SAM2)
for generating high-quality segmentation masks from bounding box detections,
enabling pixel-level object identification in volleyball game analysis.
"""
import numpy as np

SAM2_AVAILABLE = False
try:
    import torch
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM2_AVAILABLE = True
    except ImportError:
        from segment_anything_2.sam2.build_sam import build_sam2
        from segment_anything_2.sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM2_AVAILABLE = True
except ImportError:
    pass


class SAM2Segmentor:
    """
    SAM2-based object segmentation using bounding box prompts.

    This class wraps the SAM2 model to provide object segmentation capabilities
    given bounding box detections. It converts coarse bounding boxes into precise
    pixel-level segmentation masks for enhanced object analysis.
    """

    def __init__(self, checkpoint_path, config_path, device='cuda'):
        """
        Initialize the SAM2 segmentation model.

        Loads the SAM2 model from checkpoint and configuration files, preparing
        it for segmentation inference on the specified device.

        Args:
            checkpoint_path (str): Path to the SAM2 model checkpoint file containing
                the trained weights.
            config_path (str): Path to the SAM2 configuration file specifying model
                architecture and parameters.
            device (str, optional): Device to run the model on, either 'cuda' for GPU
                or 'cpu'. Defaults to 'cuda' for optimal performance.
        """
        self.device = device
        self.sam2_model = build_sam2(config_path, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        print(f"SAM2 loaded from {checkpoint_path}")

    def segment_objects(self, frame, detections):
        """
        Generate segmentation masks for all detected objects in a frame.

        Takes a list of bounding box detections and produces corresponding
        segmentation masks by prompting the SAM2 model with each box. The method
        handles errors gracefully, returning None for detections that fail to segment.

        Args:
            frame (numpy.ndarray): Input image frame in RGB or BGR format with
                shape (H, W, 3).
            detections (list): List of detections where each detection is at minimum
                [x1, y1, x2, y2] or longer with additional fields like confidence.

        Returns:
            list: List of segmentation masks corresponding to each detection, where
                each mask is a boolean numpy.ndarray of shape (H, W) with True values
                indicating the object region. Returns None for individual detections
                that fail to segment. Returns empty list if no detections are provided.
        """
        if len(detections) == 0:
            return []
        self.predictor.set_image(frame)
        masks = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            input_box = np.array([x1, y1, x2, y2])
            try:
                mask_outputs, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                mask = mask_outputs[0].astype(bool)
                masks.append(mask)
            except Exception as e:
                masks.append(None)
        return masks
