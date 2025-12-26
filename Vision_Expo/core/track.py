"""
Track object for maintaining individual object state across frames.

This module defines the Track class which represents a single tracked object
through time, maintaining its state, motion history, and predictions using
Kalman filtering for smooth trajectory estimation.
"""
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter
from utils.geometry import centroid, box_area


class Track:
    """
    Individual track object with Kalman filtering and state management.

    This class maintains the complete state of a tracked object including its position,
    velocity, acceleration, appearance history, team classification, and stability metrics.
    It uses a Kalman filter to predict future positions and smooth trajectories.
    """

    def __init__(self, tid, bbox, frame_id, track_type='person'):
        """
        Initialize a new track with initial detection and configuration.

        Args:
            tid (int): Unique track identifier assigned by the tracker.
            bbox (list or numpy.ndarray): Initial bounding box as [x1, y1, x2, y2]
                where (x1, y1) is top-left and (x2, y2) is bottom-right corner.
            frame_id (int): Frame identifier where this track is first created.
            track_type (str, optional): Type of tracked object, either 'person' or 'ball'.
                Affects Kalman filter parameters and tracking behavior. Defaults to 'person'.
        """
        self.id = tid
        self.bbox = np.array(bbox)
        self.centroid = centroid(bbox)
        self.type = track_type
        self.age = 0
        self.hits = 1
        self.misses = 0
        self.last_seen = frame_id
        self.confirmed = False
        self.created_frame = frame_id

        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.prev_velocity = np.zeros(2)

        self.history = deque(maxlen=30)
        self.history.append(self.centroid.copy())
        self.bbox_history = deque(maxlen=10)
        self.bbox_history.append(bbox)

        self.team = None
        self.is_ref = False
        self.ref_confidence = 0.0
        self.team_votes = deque(maxlen=15)

        self.mask = None
        self.mask_history = deque(maxlen=5)

        if track_type == 'ball':
            self.trajectory = deque(maxlen=100)
            self.trajectory.append(self.centroid.copy())
            self.velocity_3d = None

        self.position_variance = 0.0
        self.size_variance = 0.0

        self._init_kalman()

    def _init_kalman(self):
        """
        Initialize the Kalman filter for position and velocity prediction.

        Creates a 6-dimensional state space Kalman filter that tracks position,
        velocity, and acceleration in both x and y dimensions. The filter parameters
        (process noise, measurement noise, and initial covariance) are tuned differently
        for person and ball tracking based on their typical motion characteristics.

        The state vector is [x, y, vx, vy, ax, ay] where:
            - x, y: Position coordinates
            - vx, vy: Velocity components
            - ax, ay: Acceleration components
        """
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        self.kf.x = np.array([self.centroid[0], self.centroid[1], 0, 0, 0, 0])

        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        if self.type == 'ball':
            self.kf.P *= 50
            self.kf.R *= 0.5
            self.kf.Q *= 2.0
        else:
            self.kf.P *= 10
            self.kf.R *= 2
            self.kf.Q *= 0.05

    def predict(self):
        """
        Predict the next position using the Kalman filter.

        This method advances the Kalman filter state by one time step using the
        motion model, providing a predicted position for the next frame. This
        prediction is used during the association phase to match detections with
        existing tracks.

        Returns:
            numpy.ndarray: Predicted centroid position as [x, y] for the next frame.
        """
        self.kf.predict()
        return self.kf.x[:2]

    def update(self, bbox, frame_id, mask=None):
        """
        Update the track state with a new detection measurement.

        This method processes a new detection that has been associated with this track,
        updating all relevant state information including position, velocity, acceleration,
        history buffers, and Kalman filter state. It also updates track confirmation
        status and stability metrics.

        Args:
            bbox (list or numpy.ndarray): New bounding box as [x1, y1, x2, y2].
            frame_id (int): Frame identifier for this update.
            mask (numpy.ndarray, optional): Segmentation mask for this detection,
                typically a boolean array matching frame dimensions. Defaults to None.
        """
        new_cent = centroid(bbox)

        self.prev_velocity = self.velocity.copy()
        self.velocity = new_cent - self.centroid
        self.acceleration = self.velocity - self.prev_velocity

        self.kf.update(new_cent)

        self.centroid = new_cent
        self.bbox = np.array(bbox)
        self.history.append(new_cent.copy())
        self.bbox_history.append(bbox)

        if self.type == 'ball' and hasattr(self, 'trajectory'):
            self.trajectory.append(new_cent.copy())

        if mask is not None:
            self.mask = mask
            self.mask_history.append(mask)

        self.hits += 1
        self.misses = 0
        self.age += 1
        self.last_seen = frame_id

        if self.hits >= 3 and self.age >= 3:
            self.confirmed = True

        self._update_stability()

    def mark_missed(self):
        """
        Mark this track as having no associated detection in the current frame.

        This method is called when the track could not be matched to any detection
        in the current frame. It increments the miss counter and age, which are used
        to determine when the track should be deleted.
        """
        self.misses += 1
        self.age += 1

    def _update_stability(self):
        """
        Calculate and update stability metrics for the track.

        Computes variance measures for both position and size history to assess
        how stable and reliable this track is. These metrics help identify tracks
        that are consistently detected versus those that are intermittent or erratic.
        Requires at least 5 frames of history for meaningful variance calculation.
        """
        if len(self.history) < 5:
            return

        recent_positions = np.array(list(self.history)[-10:])
        self.position_variance = np.var(recent_positions, axis=0).mean()

        if len(self.bbox_history) >= 5:
            areas = [box_area(b) for b in list(self.bbox_history)[-5:]]
            self.size_variance = np.var(areas)

    def get_predicted_bbox(self):
        """
        Generate a predicted bounding box for the next frame.

        Uses the Kalman filter's predicted centroid position combined with the
        current bounding box dimensions to estimate where the object will be
        in the next frame. This predicted box is used during the detection
        association phase.

        Returns:
            numpy.ndarray: Predicted bounding box as [x1, y1, x2, y2] where the
                box is centered on the predicted centroid with the same width and
                height as the current bounding box.
        """
        pred_cent = self.kf.x[:2]
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return np.array([
            pred_cent[0] - w/2,
            pred_cent[1] - h/2,
            pred_cent[0] + w/2,
            pred_cent[1] + h/2
        ])

    def is_stable(self):
        """
        Determine if this track is stable and reliable.

        A track is considered stable if it has existed for at least 5 frames,
        has low position variance indicating consistent detections, and has been
        successfully matched (hits) more than twice as often as it has been missed.

        Returns:
            bool: True if the track meets all stability criteria, False otherwise.
                Young tracks (age < 5) always return False regardless of other metrics.
        """
        if self.age < 5:
            return False
        return self.position_variance < 50 and self.hits > self.misses * 2
