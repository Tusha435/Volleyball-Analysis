"""
Basic geometric operations for bounding box calculations.
"""
import numpy as np


def centroid(bbox):
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: Array-like [x1, y1, x2, y2] representing box corners

    Returns:
        numpy.ndarray: Center coordinates [cx, cy]
    """
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def iou(box1, box2):
    """
    Compute intersection over union between two boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        float: IoU score in range [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-8)


def box_area(bbox):
    """
    Calculate the area of a bounding box.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        float: Area in pixels
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
