"""
Position-locked referee classification system for volleyball court analysis.

This module provides a strict position-based approach to referee detection,
where referee identities are determined by their proximity to manually specified
court positions. The system maintains position locks for each expected referee
and assigns tracks to these positions based on spatial and temporal criteria.
"""
import numpy as np


class PositionLockedRefereeClassifier:
    """
    Manages referee detection using fixed court positions with position locking.

    This classifier implements a strict spatial approach where only player tracks
    detected near predefined referee positions are classified as referees. Each
    referee position maintains a locking mechanism that tracks assignments across
    frames, enabling robust referee identification even with camera motion.

    The system uses homography transformations to compensate for camera movement,
    ensuring that referee positions remain accurately anchored to the court as
    the camera pans or shifts.

    Attributes:
        h: Height of the video frame in pixels.
        w: Width of the video frame in pixels.
        scale: Optional scale adapter for resolution-independent distance calculations.
        manual_positions: List of manually specified referee positions on the court.
        position_locks: Dictionary mapping referee IDs to their position lock data,
            including target position, detection radius, locked track ID, lock
            strength, and strike count for handling temporary disappearances.
    """

    def __init__(self, frame_shape, manual_ref_positions, scale_adapter=None):
        """
        Initializes the referee classifier with court dimensions and referee positions.

        Args:
            frame_shape: Tuple containing (height, width) of the video frame.
            manual_ref_positions: List of (x, y) coordinate tuples indicating where
                referees are expected to be positioned on the court.
            scale_adapter: Optional adapter object for scaling distances based on
                frame resolution. If None, uses default pixel distances.
        """
        self.h, self.w = frame_shape[:2]
        self.scale = scale_adapter
        self.manual_positions = manual_ref_positions
        self.position_locks = {}

        for i, pos in enumerate(manual_ref_positions):
            ref_id = i + 1
            self.position_locks[ref_id] = {
                'position': np.array(pos),
                'radius': self.scale.scale_distance(260) if self.scale else 260,
                'locked_track_id': None,
                'lock_strength': 0,
                'strike_count': 0
            }

        print(f"✓ Position locks: {len(self.position_locks)} referee zones")

    def find_referee_for_position(self, ref_id, available_tracks):
        """
        Identifies the best track candidate for a specific referee position.

        This method evaluates all available tracks and selects the one that best
        matches the expected referee position based on multiple criteria including
        proximity, stability, position history, and track maturity.

        Args:
            ref_id: Integer identifier for the referee position to fill.
            available_tracks: Dictionary mapping track IDs to Track objects,
                representing all currently tracked entities in the frame.

        Returns:
            Tuple of (track_id, track_object, distance, score) for the best candidate,
            or None if no suitable track is found within the search radius.
        """
        if ref_id not in self.position_locks:
            return None

        lock = self.position_locks[ref_id]
        target_pos = lock['position']
        radius = lock['radius']

        candidates = []
        for tid, track in available_tracks.items():
            if not track.confirmed or track.hits < 8:
                continue

            dist = np.linalg.norm(track.centroid - target_pos)
            if dist < radius * 2.0:
                score = 0.0

                score += (1.0 - min(dist / radius, 1.0)) * 10.0

                if track.is_stable():
                    score += 3.0

                if len(track.history) >= 15:
                    recent = list(track.history)[-15:]
                    avg_pos = np.mean(recent, axis=0)
                    avg_dist = np.linalg.norm(avg_pos - target_pos)
                    if avg_dist < radius:
                        score += 5.0

                if track.age > 30:
                    score += 2.0

                candidates.append((tid, track, dist, score))

        if len(candidates) == 0:
            return None

        candidates.sort(key=lambda x: x[3], reverse=True)
        return candidates[0]

    def is_referee_at_position(self, track, ref_id):
        """
        Determines whether a track is located at a specific referee position.

        Args:
            track: Track object to evaluate.
            ref_id: Integer identifier for the referee position to check against.

        Returns:
            Boolean indicating whether the track's centroid falls within the
            detection radius of the specified referee position.
        """
        if ref_id not in self.position_locks:
            return False

        lock = self.position_locks[ref_id]
        dist = np.linalg.norm(track.centroid - lock['position'])
        return dist < lock['radius']

    def update_anchor(self, ref_id, homography):
        """
        Adjusts a referee position anchor to compensate for camera motion.

        This method applies a homography transformation to update the stored
        position of a referee zone, allowing the system to maintain accurate
        position locks even when the camera pans, tilts, or shifts during recording.

        Args:
            ref_id: Integer identifier for the referee position to update.
            homography: 3x3 numpy array representing the homography transformation
                matrix from the previous frame to the current frame. If None, no
                update is performed.
        """
        if homography is None or ref_id not in self.position_locks:
            return

        pos = self.position_locks[ref_id]['position']
        pt = np.array([[pos[0], pos[1], 1.0]]).T
        new_pt = homography @ pt
        new_pt /= new_pt[2]

        if np.any(np.isnan(new_pt)) or np.any(np.isinf(new_pt)):
            return
        new_x, new_y = new_pt[0, 0], new_pt[1, 0]
        if new_x < 0 or new_x >= self.w or new_y < 0 or new_y >= self.h:
            return

        self.position_locks[ref_id]['position'] = np.array([new_x, new_y])
