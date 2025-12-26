"""
Enhanced object detection module using YOLO for volleyball analysis.

This module provides a wrapper around the YOLO object detection model with custom
validation and filtering logic specific to volleyball scenarios, including person
and ball detection with configurable confidence thresholds.
"""
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Install ultralytics")
    exit(1)


class EnhancedDetector:
    """
    Enhanced object detector with custom validation for volleyball scenarios.

    This detector wraps YOLO and applies additional filtering and validation
    specific to volleyball game analysis, including size constraints and
    aspect ratio checks for both person and ball detections.
    """

    def __init__(self, model_path, device, conf_person, conf_ball, error_logger=None, scale_adapter=None):
        """
        Initialize the enhanced detector with YOLO model and configuration.

        Args:
            model_path (str): Path to the YOLO model weights file.
            device (str): Device to run inference on ('cuda' or 'cpu').
            conf_person (float): Confidence threshold for person detections (0.0-1.0).
            conf_ball (float): Confidence threshold for ball detections (0.0-1.0).
            error_logger (Logger, optional): Logger instance for error reporting.
            scale_adapter (ScaleAdapter, optional): Adapter for scaling distance measurements
                based on camera perspective and field dimensions.
        """
        self.model = YOLO(model_path)
        self.device = device
        self.conf_person = conf_person
        self.conf_ball = conf_ball
        self.error_logger = error_logger
        self.scale = scale_adapter
        self.prev_person_count = 0
        self.prev_ball_count = 0

    def detect(self, frame, frame_id):
        """
        Perform object detection on a single frame with validation and filtering.

        This method runs YOLO inference and applies custom validation logic to filter
        out invalid detections based on size constraints, aspect ratios, and other
        heuristics specific to volleyball scenarios.

        Args:
            frame (numpy.ndarray): Input image frame in BGR format with shape (H, W, 3).
            frame_id (int): Unique identifier for the current frame.

        Returns:
            tuple: A 3-element tuple containing:
                - persons (list): List of validated person detections, each as
                    [x1, y1, x2, y2, confidence] where coordinates are in pixels.
                - balls (list): List of validated ball detections, each as
                    [x1, y1, x2, y2, confidence] where coordinates are in pixels.
                - quality (dict): Dictionary containing detection quality metrics:
                    - 'person_detections': Number of valid person detections
                    - 'ball_detections': Number of valid ball detections
                    - 'filtered_detections': Number of detections filtered out
                    - 'suspicious_detections': Number of suspicious detections found
        """
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=min(self.conf_person, self.conf_ball),
            device=self.device,
            verbose=False,
            iou=0.5
        )
        persons = []
        balls = []
        quality = {
            'person_detections': 0,
            'ball_detections': 0,
            'filtered_detections': 0,
            'suspicious_detections': 0
        }

        if results and len(results) > 0:
            boxes = results[0].boxes
            person_boxes = []
            ball_boxes = []

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                if cls == 0 and conf >= self.conf_person:
                    person_boxes.append([x1, y1, x2, y2, conf])
                elif cls == 32 and conf >= self.conf_ball:
                    ball_boxes.append([x1, y1, x2, y2, conf])

            persons = self._apply_custom_nms(person_boxes, iou_threshold=0.4)

            validated_persons = []
            for person in persons:
                x1, y1, x2, y2, conf = person
                pw = x2 - x1
                ph = y2 - y1

                min_person_width = self.scale.scale_distance(20) if self.scale else 20
                min_person_height = self.scale.scale_distance(40) if self.scale else 40
                max_person_size = min(w, h) * 0.8

                if pw < min_person_width or ph < min_person_height:
                    quality['filtered_detections'] += 1
                    continue
                if pw > max_person_size or ph > max_person_size:
                    quality['suspicious_detections'] += 1
                    continue

                aspect = max(pw, ph) / (min(pw, ph) + 1e-8)
                if aspect > 4.0:
                    quality['filtered_detections'] += 1
                    continue

                validated_persons.append(person)

            persons = validated_persons

            validated_balls = []
            for ball in ball_boxes:
                x1, y1, x2, y2, conf = ball
                bw = x2 - x1
                bh = y2 - y1

                min_ball_size = self.scale.scale_distance(8) if self.scale else 8
                max_ball_size = self.scale.scale_distance(60) if self.scale else 60

                if bw < min_ball_size or bh < min_ball_size:
                    continue
                if bw > max_ball_size or bh > max_ball_size:
                    continue

                validated_balls.append(ball)

            balls = validated_balls

        quality['person_detections'] = len(persons)
        quality['ball_detections'] = len(balls)
        self.prev_person_count = len(persons)
        self.prev_ball_count = len(balls)

        return persons, balls, quality

    def _apply_custom_nms(self, boxes, iou_threshold=0.4):
        """
        Apply custom Non-Maximum Suppression to remove overlapping detections.

        This method implements the NMS algorithm to eliminate redundant bounding boxes
        by keeping only the highest-confidence detection when multiple boxes overlap
        significantly based on their Intersection over Union (IoU) score.

        Args:
            boxes (list): List of bounding boxes, each as [x1, y1, x2, y2, confidence]
                where (x1, y1) is top-left corner and (x2, y2) is bottom-right corner.
            iou_threshold (float, optional): IoU threshold above which boxes are
                considered overlapping and should be suppressed. Defaults to 0.4.

        Returns:
            list: Filtered list of bounding boxes after NMS, maintaining the same
                format as input with only non-overlapping high-confidence detections.
        """
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou_vals = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            inds = np.where(iou_vals <= iou_threshold)[0]
            order = order[inds + 1]

        return boxes[keep].tolist()
