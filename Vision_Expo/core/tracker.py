"""
Multi-object tracking module implementing Hungarian algorithm for data association.

This module provides robust tracking capabilities for volleyball game analysis,
using the Hungarian algorithm for optimal detection-to-track assignment and
maintaining track state across frames with sophisticated consistency checks.
"""
import numpy as np
from collections import OrderedDict, deque
from scipy.optimize import linear_sum_assignment
from .track import Track
from utils.geometry import centroid, iou, box_area


class Tracker:
    """
    Multi-object tracker using Hungarian algorithm for optimal assignment.

    This tracker maintains multiple object tracks over time, associating new detections
    with existing tracks using a cost matrix based on IoU, distance, and size similarity.
    It handles track creation, update, and deletion with sophisticated consistency checks.
    """

    def __init__(self, max_age, iou_thresh, track_type='person', error_logger=None, scale_adapter=None):
        """
        Initialize the multi-object tracker with configuration parameters.

        Args:
            max_age (int): Maximum number of consecutive frames a track can be missed
                before being deleted from the active track list.
            iou_thresh (float): Minimum Intersection over Union threshold for considering
                a detection-track pair as a potential match (0.0-1.0).
            track_type (str, optional): Type of objects being tracked, either 'person'
                or 'ball'. Affects tracking behavior. Defaults to 'person'.
            error_logger (Logger, optional): Logger instance for error reporting.
            scale_adapter (ScaleAdapter, optional): Adapter for scaling distance measurements
                based on camera perspective and field dimensions.
        """
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.track_type = track_type
        self.tracks = OrderedDict()
        self.next_id = 1
        self.error_logger = error_logger
        self.scale = scale_adapter
        self.deleted_tracks = deque(maxlen=50)
        self.track_id_map = {}
        self.allow_new_tracks = True

    def update(self, detections, frame_id, homography=None):
        """
        Update all tracks with new detections for the current frame.

        This method performs the complete tracking update cycle: predicting current
        positions for all existing tracks, computing a cost matrix between tracks and
        detections, solving the assignment problem using the Hungarian algorithm,
        updating matched tracks, marking unmatched tracks as missed, creating new tracks
        for unmatched detections, and removing old tracks.

        Args:
            detections (list): List of detections for the current frame, where each
                detection is [x1, y1, x2, y2, confidence] or [x1, y1, x2, y2, confidence, mask].
            frame_id (int): Unique identifier for the current frame.
            homography (numpy.ndarray, optional): Homography matrix for perspective
                transformation. Currently unused but reserved for future enhancements.
        """
        for track in self.tracks.values():
            track.predict()

        if len(detections) == 0:
            for track in self.tracks.values():
                track.mark_missed()
            self._cleanup(frame_id)
            return

        track_objs = list(self.tracks.values())
        cost = self._compute_cost_matrix(track_objs, detections, homography)

        if cost.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind = col_ind = []

        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 0.65:
                tid = track_ids[i]
                track = self.tracks[tid]
                if self._is_consistent_match(track, detections[j][:4]):
                    mask = detections[j][5] if len(detections[j]) > 5 else None
                    track.update(detections[j][:4], frame_id, mask)
                    matched_tracks.add(tid)
                    matched_dets.add(j)

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].mark_missed()

        if self.allow_new_tracks:
            for j, det in enumerate(detections):
                if j not in matched_dets:
                    self._create_track(det, frame_id)

        self._cleanup(frame_id)

    def _compute_cost_matrix(self, tracks, detections, homography):
        """
        Compute the cost matrix for Hungarian algorithm assignment.

        The cost matrix combines multiple similarity metrics including Intersection
        over Union (IoU), centroid distance, and bounding box size similarity to
        create a robust association between existing tracks and new detections.

        Args:
            tracks (list): List of Track objects representing active tracks.
            detections (list): List of detections for the current frame, where each
                detection is [x1, y1, x2, y2, confidence] or longer with optional mask.
            homography (numpy.ndarray, optional): Homography matrix for perspective
                transformation. Currently unused but reserved for future enhancements.

        Returns:
            numpy.ndarray: Cost matrix of shape (n_tracks, n_detections) where each
                element [i, j] represents the cost of associating track i with detection j.
                Lower costs indicate better matches. Returns empty array if either
                tracks or detections is empty.
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.array([])

        n_tracks = len(tracks)
        n_dets = len(detections)
        cost = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(tracks):
            pred_bbox = track.get_predicted_bbox()
            for j, det in enumerate(detections):
                det_bbox = det[:4]

                iou_score = iou(pred_bbox, det_bbox)

                pred_cent = centroid(pred_bbox)
                det_cent = centroid(det_bbox)
                dist = np.linalg.norm(pred_cent - det_cent)
                max_dist = self.scale.scale_distance(200) if self.scale else 200
                dist_score = max(0, 1.0 - dist / max_dist)

                pred_area = box_area(pred_bbox)
                det_area = box_area(det_bbox)
                size_ratio = min(pred_area, det_area) / (max(pred_area, det_area) + 1e-8)

                combined_score = (iou_score * 0.5 + dist_score * 0.3 + size_ratio * 0.2)
                cost[i, j] = 1.0 - combined_score

        return cost

    def _is_consistent_match(self, track, det_bbox):
        """
        Verify if a detection is consistent with a track's motion history.

        This method checks whether a proposed detection matches the expected motion
        pattern of a track by comparing the detection position with the predicted
        position based on the track's velocity. This helps prevent incorrect
        associations during occlusions or when objects pass close to each other.

        Args:
            track (Track): The track object to check consistency against.
            det_bbox (list or numpy.ndarray): Detection bounding box as [x1, y1, x2, y2].

        Returns:
            bool: True if the detection is consistent with the track's expected motion,
                False otherwise. Always returns True for young tracks (age < 3) since
                they don't have enough history for reliable motion prediction.
        """
        if track.age < 3:
            return True

        det_cent = centroid(det_bbox)
        expected_cent = track.centroid + track.velocity
        dist = np.linalg.norm(det_cent - expected_cent)
        max_dist = self.scale.scale_distance(150) if self.scale else 150

        if dist > max_dist:
            return False

        return True

    def _create_track(self, detection, frame_id):
        """
        Create a new track from an unmatched detection.

        This method initializes a new Track object with a unique ID and adds it
        to the active tracks dictionary. If the detection includes a segmentation
        mask, it is also stored with the track.

        Args:
            detection (list): Detection data as [x1, y1, x2, y2, confidence] or
                [x1, y1, x2, y2, confidence, mask] where mask is optional.
            frame_id (int): Frame identifier where this track is first created.
        """
        new_track = Track(self.next_id, detection[:4], frame_id, self.track_type)
        if len(detection) > 5:
            new_track.mask = detection[5]
        self.tracks[self.next_id] = new_track
        self.next_id += 1

    def _cleanup(self, frame_id):
        """
        Remove tracks that have exceeded the maximum age without detections.

        This method identifies tracks that haven't been matched to any detection
        for more than max_age consecutive frames and removes them from the active
        tracks dictionary. Deleted tracks are moved to a history buffer for
        potential future analysis or recovery.

        Args:
            frame_id (int): Current frame identifier used for tracking deletion timing.
        """
        to_remove = []
        for tid, track in self.tracks.items():
            if track.misses > self.max_age:
                to_remove.append(tid)

        for tid in to_remove:
            self.deleted_tracks.append((tid, frame_id, self.tracks[tid]))
            del self.tracks[tid]
