"""
Advanced volleyball tracking system with multi-object detection and tracking.

This module provides the VolleyballTrackerPro class which orchestrates player detection,
ball tracking, referee identification, and team assignment for volleyball game analysis.
It integrates computer vision techniques including object detection, optical flow tracking,
and optional SAM2 segmentation for robust multi-object tracking in volleyball scenarios.
"""
import cv2
import numpy as np
from config import CONFIG
from core import EnhancedDetector, Tracker, SAM2Segmentor, SAM2_AVAILABLE
from classifiers import PositionLockedRefereeClassifier, TeamAssigner
from utils import (ErrorLogger, ScaleAdapter, CameraMotionCompensator,
                   OpticalFlowNetTracker, NetTracker, iou)


class VolleyballTrackerPro:
    """
    Professional volleyball tracking system with intelligent object detection and tracking.

    This class manages the complete volleyball tracking pipeline, including player detection,
    ball tracking, referee identification, team assignment, and camera motion compensation.
    It uses YOLO-based detection combined with custom tracking algorithms optimized for
    volleyball game scenarios.

    Attributes:
        error_logger: Logging system for tracking errors and debugging
        scale_adapter: Handles coordinate scaling across different video resolutions
        detector: Enhanced YOLO detector for players and balls
        player_tracker: Tracker instance for player movement
        ref_tracker: Dedicated tracker for referee positions
        ball_tracker: Specialized tracker for ball detection
        ref_classifier: Position-based classifier for referee identification
        team_assigner: System for assigning players to teams based on court position
        net_tracker: Tracks the volleyball net position
        net_flow_tracker: Optical flow-based net position tracker
        camera_compensator: Compensates for camera movement using homography
        sam2: Optional SAM2 segmentation model for precise object masks
        frame_count: Current frame number in processing
        manual_ref_positions: User-selected referee position anchors
    """

    def __init__(self):
        """Initialize the volleyball tracker with empty component slots."""
        self.error_logger = ErrorLogger(CONFIG['log_errors'])
        self.scale_adapter = None
        self.detector = None
        self.player_tracker = None
        self.ref_tracker = None
        self.ball_tracker = None
        self.ref_classifier = None
        self.team_assigner = TeamAssigner()
        self.net_tracker = None
        self.net_flow_tracker = None
        self.camera_compensator = None
        self.sam2 = None
        self.frame_count = 0
        self.manual_ref_positions = []

    def setup(self, first_frame):
        """
        Initialize all tracking components using the first video frame.

        This method performs critical setup including detector initialization, tracker
        configuration, manual net and referee selection, and optional SAM2 segmentation
        initialization. The user must manually select the net endpoints and referee
        positions during this phase.

        Args:
            first_frame: The first frame from the video stream as a numpy array in BGR format.
                        Used for initialization and manual selection of reference points.

        Returns:
            bool: True if setup completed successfully, False if user cancelled setup
                 or initialization failed.
        """
        h, w = first_frame.shape[:2]
        self.scale_adapter = ScaleAdapter(w, h, CONFIG['reference_width'], CONFIG['reference_height'])

        self.detector = EnhancedDetector(
            CONFIG['model_path'], CONFIG['device'],
            CONFIG['conf_person'], CONFIG['conf_ball'],
            self.error_logger, self.scale_adapter
        )

        self.player_tracker = Tracker(
            CONFIG['max_age_player'], CONFIG['iou_player'],
            'person', self.error_logger, self.scale_adapter
        )
        self.ref_tracker = Tracker(
            CONFIG['max_age_ref'], CONFIG['iou_ref'],
            'referee', self.error_logger, self.scale_adapter
        )
        self.ref_tracker.allow_new_tracks = False

        self.ball_tracker = Tracker(
            CONFIG['max_age_ball'], CONFIG['iou_ball'],
            'ball', self.error_logger, self.scale_adapter
        )

        if CONFIG['use_homography']:
            self.camera_compensator = CameraMotionCompensator(
                CONFIG['feature_detection'], self.scale_adapter
            )

        net_pts, ref_pts = self._manual_select(first_frame)
        if net_pts is None:
            return False

        net_dense_pts = self._densify_line(net_pts[0], net_pts[1], num=7)
        self.net_flow_tracker = OpticalFlowNetTracker(net_dense_pts, first_frame, self.scale_adapter)

        self.manual_ref_positions = ref_pts if ref_pts else []
        self.ref_classifier = PositionLockedRefereeClassifier(
            first_frame.shape, self.manual_ref_positions, self.scale_adapter
        )

        self.team_assigner.set_net(net_pts[0], net_pts[1])
        self.net_tracker = NetTracker(net_dense_pts, first_frame.shape)

        if self.camera_compensator:
            self.camera_compensator.update(first_frame)

        if CONFIG['use_sam2'] and SAM2_AVAILABLE and CONFIG['sam2_checkpoint']:
            try:
                self.sam2 = SAM2Segmentor(
                    CONFIG['sam2_checkpoint'],
                    CONFIG['sam2_config'],
                    CONFIG['device']
                )
                print("✓ SAM2 segmentation enabled")
            except Exception as e:
                print(f"⚠ SAM2 initialization failed: {e}")
                self.sam2 = None

        return True

    def _manual_select(self, frame):
        """
        Interactive manual selection of net endpoints and referee positions.

        Provides a graphical interface for the user to click on the volleyball net
        endpoints and referee positions. The interface displays real-time feedback
        with visual overlays showing selected points and referee zones.

        Args:
            frame: Video frame to display for manual selection, numpy array in BGR format.

        Returns:
            tuple: A tuple of (net_pts, ref_pts) where:
                - net_pts: List of two (x, y) tuples marking net endpoints
                - ref_pts: List of two (x, y) tuples marking referee positions
                Returns (None, None) if user cancels with Escape key.
        """
        clone = frame.copy()
        net_pts = []
        ref_pts = []
        mode = "net"

        def mouse(event, x, y, flags, param):
            nonlocal mode
            if event == cv2.EVENT_LBUTTONDOWN:
                if mode == "net" and len(net_pts) < 2:
                    net_pts.append((x, y))
                elif mode == "ref" and len(ref_pts) < 2:
                    ref_pts.append((x, y))

        cv2.namedWindow("Setup", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Setup", mouse)

        while True:
            vis = clone.copy()

            for p in net_pts:
                cv2.circle(vis, p, 10, (0, 255, 255), -1)
            if len(net_pts) == 2:
                cv2.line(vis, net_pts[0], net_pts[1], (0, 255, 255), 4)

            if mode == "ref":
                for i, p in enumerate(ref_pts):
                    cv2.circle(vis, p, 8, (255, 0, 255), -1)
                    zone_radius = int(self.scale_adapter.scale_distance(80))
                    cv2.circle(vis, p, zone_radius, (255, 0, 255), 2)
                    cv2.putText(vis, f"REF {i+1}", (p[0] - 30, p[1] - zone_radius - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            if mode == "net":
                msg = "Click 2 NET endpoints - ENTER to continue"
                color = (0, 255, 255)
            else:
                msg = f"Click EXACTLY 2 REFEREE positions ({len(ref_pts)}/2) - ENTER when done"
                color = (255, 0, 255)

            cv2.putText(vis, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Setup", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                if mode == "net" and len(net_pts) == 2:
                    mode = "ref"
                elif mode == "ref" and len(ref_pts) == 2:
                    break
            elif key == 27:
                cv2.destroyAllWindows()
                return None, None

        cv2.destroyAllWindows()
        print(f"✓ Setup complete: {len(net_pts)} net points, {len(ref_pts)} referee positions")
        return net_pts, ref_pts

    def _densify_line(self, p1, p2, num=7):
        """
        Generate evenly-spaced points along a line segment.

        Creates intermediate points between two endpoints for optical flow tracking.
        More points provide better tracking stability for the volleyball net.

        Args:
            p1: First endpoint as (x, y) tuple
            p2: Second endpoint as (x, y) tuple
            num: Number of points to generate along the line (default: 7)

        Returns:
            list: List of (x, y) tuples representing evenly-spaced points from p1 to p2.
        """
        return [
            (int(p1[0] + i * (p2[0] - p1[0]) / (num - 1)),
             int(p1[1] + i * (p2[1] - p1[1]) / (num - 1)))
            for i in range(num)
        ]

    def _enforce_single_ball(self, frame_id):
        """
        Ensure only the most confident ball track remains active.

        In volleyball tracking, only one ball should be tracked at a time. This method
        evaluates all ball tracks using a scoring system based on detection consistency,
        trajectory smoothness, and position variance, then removes all but the best track.

        Args:
            frame_id: Current frame number, used for temporal context in scoring.
        """
        balls = list(self.ball_tracker.tracks.values())
        if len(balls) <= 1:
            return

        scored = []
        for b in balls:
            score = 0.0
            score += b.hits * 2.0
            score -= b.misses * 3.0
            if hasattr(b, "trajectory"):
                score += len(b.trajectory) * 0.5
            if b.position_variance < 30:
                score += 5.0
            scored.append((score, b))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_ball = scored[0][1]

        for _, ball in scored[1:]:
            del self.ball_tracker.tracks[ball.id]

    def process_frame(self, frame, frame_id):
        """
        Process a single video frame through the complete tracking pipeline.

        This is the main processing method that orchestrates all tracking components.
        It handles camera motion compensation, object detection, SAM2 segmentation,
        referee/player separation, track updates, and team assignment. The method
        ensures proper coordination between all subsystems for accurate tracking.

        Args:
            frame: Current video frame as numpy array in BGR format
            frame_id: Sequential frame number for temporal tracking and logging
        """
        self.frame_count = frame_id
        homography = None
        camera_stable = True

        if self.camera_compensator:
            homography, camera_stable, _ = self.camera_compensator.update(frame)

            if homography is not None:
                for ref_id in self.ref_classifier.position_locks:
                    self.ref_classifier.update_anchor(ref_id, homography)

            if self.net_flow_tracker:
                net_pts_flow = self.net_flow_tracker.update(frame)
                flow_stable = self.net_flow_tracker.stable

            if self.net_tracker:
                if flow_stable:
                    self.net_tracker.current_points = net_pts_flow
                    self.net_tracker.stable = True
                    self.net_tracker.confidence = self.net_flow_tracker.confidence
                else:
                    self.net_tracker.update(homography, camera_stable)

        persons, balls, det_quality = self.detector.detect(frame, frame_id)

        person_masks = []
        if self.sam2 and len(persons) > 0:
            try:
                person_masks = self.sam2.segment_objects(frame, persons)
            except:
                pass

        person_dets_with_masks = []
        for i, person in enumerate(persons):
            if i < len(person_masks):
                person_dets_with_masks.append(person + [person_masks[i]])
            else:
                person_dets_with_masks.append(person + [None])

        referee_dets = []
        player_dets = []
        for person_det in person_dets_with_masks:
            is_referee_det = False
            bbox = person_det[:4]

            for ref_track in self.ref_tracker.tracks.values():
                if ref_track.confirmed:
                    ref_iou = iou(bbox, ref_track.bbox)
                    if ref_iou > 0.3:
                        is_referee_det = True
                        break

            if is_referee_det:
                referee_dets.append(person_det)
            else:
                player_dets.append(person_det)

        self.player_tracker.update(player_dets, frame_id, homography)
        self.ref_tracker.update(referee_dets, frame_id, homography)
        self.ball_tracker.update(balls, frame_id, homography)

        self._enforce_single_ball(frame_id)
        self._update_referees_strict(frame_id)

        if self.net_tracker and self.net_tracker.stable:
            p1, p2 = self.net_tracker.get_line()
            if p1 and p2:
                self.team_assigner.update_net_position(p1, p2)

        for track in self.player_tracker.tracks.values():
            if track.confirmed and not track.is_ref:
                self.team_assigner.assign(track)

    def _update_referees_strict(self, frame_id):
        """
        Maintain strict position-based referee identification with stable IDs.

        This method implements a robust referee tracking system using fixed position anchors.
        It enforces that referees must maintain IDs 1 and 2, demotes referees that drift
        from their positions, and promotes new player tracks to referee status when positions
        become vacant. Uses hysteresis to prevent flickering between referee and player states.

        Args:
            frame_id: Current frame number for temporal context in decision making.
        """
        STABLE_REF_IDS = [1, 2]

        wrong_id_refs = [tid for tid in list(self.ref_tracker.tracks.keys()) if tid not in STABLE_REF_IDS]
        for tid in wrong_id_refs:
            track = self.ref_tracker.tracks[tid]
            track.is_ref = False
            track.type = "person"
            self.player_tracker.tracks[tid] = track
            del self.ref_tracker.tracks[tid]

        existing_ref_ids = set(self.ref_tracker.tracks.keys())

        for ref_id in STABLE_REF_IDS:
            if ref_id in existing_ref_ids:
                ref_track = self.ref_tracker.tracks[ref_id]
                lock = self.ref_classifier.position_locks[ref_id]

                if not self.ref_classifier.is_referee_at_position(ref_track, ref_id):
                    lock['strike_count'] += 1
                else:
                    lock['strike_count'] = 0

                if lock['strike_count'] >= 10:
                    dist = np.linalg.norm(ref_track.centroid - lock['position'])
                    print(f"⚠ REF_{ref_id} drifted from position (dist={dist:.0f}px, strikes={lock['strike_count']}) - demoting")
                    ref_track.is_ref = False
                    ref_track.type = "person"
                    self.player_tracker.tracks[ref_id] = ref_track
                    del self.ref_tracker.tracks[ref_id]
                    existing_ref_ids.remove(ref_id)
                    lock['strike_count'] = 0

        needed_ref_ids = [rid for rid in STABLE_REF_IDS if rid not in existing_ref_ids]
        if len(needed_ref_ids) == 0:
            return

        for ref_id in needed_ref_ids:
            result = self.ref_classifier.find_referee_for_position(ref_id, self.player_tracker.tracks)
            if result is None:
                continue

            tid, track, dist, score = result
            if score < 8.0:
                continue

            duplicate = False
            for ref in self.ref_tracker.tracks.values():
                if np.linalg.norm(ref.centroid - track.centroid) < self.scale_adapter.scale_distance(80):
                    duplicate = True
                    break

            if duplicate:
                continue

            del self.player_tracker.tracks[tid]

            track.id = ref_id
            track.type = 'referee'
            track.is_ref = True
            track.confirmed = True
            track.ref_confidence = score

            self.ref_tracker.tracks[ref_id] = track
            self.ref_classifier.position_locks[ref_id]['strike_count'] = 0

            print(f"✓ Locked REF_{ref_id} (score={score:.1f}, dist={dist:.0f}px)")

    def visualize(self, frame):
        """
        Create visualization overlay showing all tracked objects and metadata.

        Generates a comprehensive visualization displaying players with team colors and IDs,
        referees with highlighted bounding boxes, ball position with trajectory, the volleyball
        net line, and an information panel showing object counts. Optionally overlays
        segmentation masks when SAM2 is enabled.

        Args:
            frame: Original video frame to overlay visualizations on, numpy array in BGR format.

        Returns:
            numpy.ndarray: Annotated frame with all tracking visualizations overlaid.
        """
        vis = frame.copy()

        if self.net_tracker and self.net_tracker.stable:
            p1, p2 = self.net_tracker.get_line()
            if p1 and p2:
                p1 = tuple(map(int, p1))
                p2 = tuple(map(int, p2))
                cv2.line(vis, p1, p2, (0, 255, 255), self.scale_adapter.scale_thickness(3))

        for track in self.player_tracker.tracks.values():
            if not track.confirmed:
                continue

            x1, y1, x2, y2 = map(int, track.bbox)
            color = (255, 100, 255) if track.team == 'A' else (100, 255, 100) if track.team == 'B' else (200, 200, 200)

            if track.mask is not None:
                try:
                    mask_colored = np.zeros_like(frame)
                    mask_colored[track.mask] = color
                    vis = cv2.addWeighted(vis, 1.0, mask_colored, 0.3, 0)
                except:
                    pass

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"P{track.id}-{track.team if track.team else '?'}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       self.scale_adapter.scale_font(0.5), color,
                       self.scale_adapter.scale_thickness(1))

        for track in self.ref_tracker.tracks.values():
            if not track.confirmed:
                continue

            x1, y1, x2, y2 = map(int, track.bbox)
            color = (0, 255, 255)

            if track.mask is not None:
                try:
                    mask_colored = np.zeros_like(frame)
                    mask_colored[track.mask] = color
                    vis = cv2.addWeighted(vis, 1.0, mask_colored, 0.3, 0)
                except:
                    pass

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            label = f"REF_{track.id}"
            font_scale = self.scale_adapter.scale_font(0.6)
            thickness = self.scale_adapter.scale_thickness(2)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(vis, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(vis, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 0, 0), thickness)

        for track in self.ball_tracker.tracks.values():
            if not track.confirmed:
                continue

            cx, cy = map(int, track.centroid)

            if hasattr(track, 'trajectory') and len(track.trajectory) > 1:
                pts = [tuple(map(int, p)) for p in list(track.trajectory)[-20:]]
                for i in range(len(pts) - 1):
                    thickness = int(self.scale_adapter.scale_thickness(2) * (i + 1) / len(pts)) + 1
                    cv2.line(vis, pts[i], pts[i+1], (0, 255, 0), thickness)

            radius = int(self.scale_adapter.scale_distance(10))
            cv2.circle(vis, (cx, cy), radius, (0, 255, 0), -1)
            cv2.circle(vis, (cx, cy), radius + 2, (255, 255, 255), 2)

        w = vis.shape[1]
        p_cnt = sum(1 for t in self.player_tracker.tracks.values() if t.confirmed)
        r_cnt = sum(1 for t in self.ref_tracker.tracks.values() if t.confirmed)
        b_cnt = sum(1 for t in self.ball_tracker.tracks.values() if t.confirmed)

        panel_width = int(self.scale_adapter.scale_distance(200))
        panel_height = int(self.scale_adapter.scale_distance(140))
        panel_x = w - panel_width - 10
        panel_y = 10

        overlay = vis.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        cv2.rectangle(vis, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

        text_x = panel_x + 15
        text_y_start = panel_y + 35
        line_spacing = int(self.scale_adapter.scale_distance(30))
        font_scale = self.scale_adapter.scale_font(0.6)
        font_thickness = self.scale_adapter.scale_thickness(2)

        cv2.putText(vis, f"Players: {p_cnt}", (text_x, text_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 200, 255), font_thickness)
        cv2.putText(vis, f"Refs: {r_cnt}", (text_x, text_y_start + line_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 100), font_thickness)
        cv2.putText(vis, f"Balls: {b_cnt}", (text_x, text_y_start + line_spacing * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 100), font_thickness)

        return vis

    def save_error_report(self):
        """
        Persist error logs to disk for debugging and analysis.

        Saves accumulated error and warning messages to the configured error log file path.
        Only saves if error logging is enabled in the configuration.
        """
        if CONFIG['log_errors']:
            self.error_logger.save(CONFIG['error_log_path'])
