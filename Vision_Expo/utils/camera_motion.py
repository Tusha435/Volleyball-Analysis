"""
Homography-based camera motion estimation for stabilizing tracking.
"""
import cv2
import numpy as np


class CameraMotionCompensator:
    """
    Estimates camera movement between frames using feature matching.

    This class uses SIFT or ORB features to detect keypoints in consecutive
    frames, matches them, and computes a homography transformation that
    describes the camera motion. This allows the tracker to compensate for
    pans, tilts, and zooms.

    Attributes:
        method: Feature detection method ('SIFT' or 'ORB')
        scale: ScaleAdapter instance for resolution compensation
        prev_gray: Previous frame in grayscale
        prev_keypoints: Keypoints from previous frame
        prev_descriptors: Feature descriptors from previous frame
        detector: OpenCV feature detector
        matcher: Feature matcher
        homography: Current homography matrix
        is_stable: Whether camera motion estimation is reliable
        confidence: Confidence score based on inlier ratio
    """

    def __init__(self, method='SIFT', scale_adapter=None):
        """
        Initialize the camera motion compensator.

        Args:
            method: Feature detection algorithm ('SIFT' or 'ORB')
            scale_adapter: ScaleAdapter instance for scaling thresholds
        """
        self.method = method
        self.scale = scale_adapter
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        if method == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.detector = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.homography = None
        self.is_stable = True
        self.confidence = 1.0

    def update(self, frame):
        """
        Compute homography between current and previous frame.

        Args:
            frame: Current BGR frame

        Returns:
            tuple: (homography_matrix, is_stable, confidence_score)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints, self.prev_descriptors = self.detector.detectAndCompute(gray, None)
            return None, True, 1.0

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None or self.prev_descriptors is None:
            self.is_stable = False
            self.confidence = 0.0
            return None, False, 0.0

        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        good_matches = []

        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            self.is_stable = False
            self.confidence = 0.0
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None, False, 0.0

        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            self.is_stable = False
            self.confidence = 0.0
        else:
            self.homography = H
            self.is_stable = True
            inliers = np.sum(mask)
            self.confidence = inliers / len(good_matches)

        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return self.homography, self.is_stable, self.confidence
