"""
Volleyball net line tracking using optical flow and homography fallback.
"""
import cv2
import numpy as np


class OpticalFlowNetTracker:
    """
    Tracks net position using Lucas-Kanade optical flow on sparse points.

    This tracker follows manually-selected net points across frames using
    pyramidal LK optical flow. It's the primary tracking method due to its
    speed and accuracy for small frame-to-frame motions.

    Attributes:
        prev_gray: Previous frame in grayscale
        points: Current tracked points along the net
        scale: ScaleAdapter for resolution-independent thresholds
        lk_params: Lucas-Kanade algorithm parameters
        stable: Whether tracking is currently reliable
        confidence: Confidence score based on motion consistency
    """

    def __init__(self, init_points, frame, scale_adapter):
        """
        Initialize optical flow tracker with net points.

        Args:
            init_points: List of (x, y) coordinates along the net
            frame: Initial BGR frame
            scale_adapter: ScaleAdapter instance
        """
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = np.array(init_points, dtype=np.float32).reshape(-1, 1, 2)
        self.scale = scale_adapter
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.stable = True
        self.confidence = 1.0

    def update(self, frame):
        """
        Track net points to the new frame using optical flow.

        Args:
            frame: Current BGR frame

        Returns:
            numpy.ndarray: Updated point positions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **self.lk_params
        )

        if new_pts is None or status.sum() < 2:
            self.stable = False
            self.confidence = 0.0
            return self.points.reshape(-1, 2)

        good_new = new_pts[status == 1]
        good_old = self.points[status == 1]
        motion = np.linalg.norm(good_new - good_old, axis=1)
        avg_motion = np.mean(motion)

        sudden_jump_threshold = self.scale.scale_distance(50)
        if avg_motion > sudden_jump_threshold:
            self.stable = False
            self.confidence = 0.3
            return self.points.reshape(-1, 2)

        self.points = new_pts
        self.prev_gray = gray
        self.stable = True
        self.confidence = max(0.1, 1.0 - avg_motion / self.scale.scale_distance(30))

        return self.points.reshape(-1, 2)

    def get_line(self):
        """
        Get the net line endpoints from tracked points.

        Returns:
            tuple: (start_point, end_point) or (None, None) if unstable
        """
        if not self.stable or len(self.points) < 2:
            return None, None
        pts = self.points.reshape(-1, 2)
        return tuple(pts[0]), tuple(pts[-1])


class NetTracker:
    """
    Homography-based net tracker as fallback when optical flow fails.

    This tracker applies the global homography transformation to the initial
    net points, providing robustness when optical flow loses tracking.

    Attributes:
        initial_points: Original net point coordinates
        current_points: Current transformed coordinates
        h: Frame height
        w: Frame width
        stable: Whether tracking is valid
        confidence: Confidence in current position
    """

    def __init__(self, initial_points, frame_shape):
        """
        Initialize homography-based net tracker.

        Args:
            initial_points: Initial net point coordinates
            frame_shape: Tuple (height, width, channels)
        """
        self.initial_points = np.array(initial_points, dtype=np.float32)
        self.current_points = self.initial_points.copy()
        self.h, self.w = frame_shape[:2]
        self.stable = True
        self.confidence = 1.0

    def update(self, homography, camera_stable):
        """
        Apply homography transformation to update net position.

        Args:
            homography: 3x3 homography matrix from camera motion
            camera_stable: Whether camera motion estimation is reliable
        """
        if homography is None or not camera_stable:
            self.stable = False
            self.confidence = 0.0
            return

        points_h = np.concatenate([self.initial_points, np.ones((len(self.initial_points), 1))], axis=1)
        transformed = (homography @ points_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]

        if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
            self.stable = False
            self.confidence = 0.0
            return

        if (np.any(transformed[:, 0] < 0) or np.any(transformed[:, 0] >= self.w) or
            np.any(transformed[:, 1] < 0) or np.any(transformed[:, 1] >= self.h)):
            self.stable = False
            self.confidence = 0.5
            return

        self.current_points = transformed
        self.stable = True
        self.confidence = 1.0

    def get_line(self):
        """
        Get the net line endpoints.

        Returns:
            tuple: (start_point, end_point) or (None, None) if unstable
        """
        if not self.stable or len(self.current_points) < 2:
            return None, None
        return tuple(self.current_points[0]), tuple(self.current_points[-1])
