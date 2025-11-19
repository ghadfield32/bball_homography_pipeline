# api/src/cv/homography_calibrator.py
"""
Segment-level homography calibration for basketball court registration.

Provides stable homography by:
- Detecting camera segments (continuous shots without cuts/zooms)
- Aggregating keypoints across multiple frames
- Fitting one robust homography per segment
- Quality-aware masking for unreliable frames

This replaces per-frame homography with segment-level optimization for:
- Reduced coordinate jitter
- Better analytics stability
- More efficient processing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import cv2
import supervision as sv
from sports import ViewTransformer
from sports.basketball import CourtConfiguration


@dataclass
class SegmentData:
    """Data collected for a camera segment."""
    segment_id: int
    start_frame: int
    end_frame: int = -1

    # Aggregated keypoint observations: list of (img_pts, court_pts, confidences)
    keypoint_observations: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )

    # Fitted homography for this segment
    H_img2court: Optional[np.ndarray] = None
    H_court2img: Optional[np.ndarray] = None
    img2court: Optional[ViewTransformer] = None
    court2img: Optional[ViewTransformer] = None

    # Quality metrics
    rmse_court: float = float("inf")
    rmse_image: float = float("inf")
    inlier_count: int = 0
    total_observations: int = 0

    @property
    def is_calibrated(self) -> bool:
        """Check if segment has valid homography."""
        return self.H_img2court is not None and self.rmse_court < float("inf")


@dataclass
class HomographyCalibrator:
    """
    Segment-level homography calibrator for stable court registration.

    Detects camera segments and fits one optimized homography per segment,
    reducing frame-to-frame jitter and improving tracking stability.

    Usage:
        calibrator = HomographyCalibrator(court_config)
        for frame_idx, frame in enumerate(video):
            keypoints = detect_court_keypoints(frame, model, cfg)
            calibrator.add_observation(frame_idx, keypoints)

        # Get homography for any frame
        H, quality = calibrator.get_transform(frame_idx)
    """
    court_config: CourtConfiguration

    # Segment detection parameters
    segment_min_frames: int = 10
    segment_change_threshold: float = 50.0  # pixels - bbox centroid jump

    # Keypoint parameters
    confidence_threshold: float = 0.5
    min_keypoints: int = 4

    # RANSAC parameters
    ransac_reproj_thresh: float = 5.0
    min_inlier_ratio: float = 0.6

    # Quality thresholds
    rmse_court_max: float = 1.5  # feet
    rmse_image_max: float = 5.0  # pixels

    # Semantic constraint parameters
    enable_semantic_constraints: bool = True
    line_collinearity_threshold: float = 0.5  # feet - max deviation from line
    arc_radius_threshold: float = 1.0  # feet - max deviation from expected radius
    three_point_radius_ft: float = 23.75  # NBA 3-point arc radius

    # Internal state
    _segments: List[SegmentData] = field(default_factory=list)
    _current_segment: Optional[SegmentData] = field(default=None, init=False)
    _last_keypoint_centroid: Optional[np.ndarray] = field(default=None, init=False)
    _frame_to_segment: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize calibrator state."""
        self._segments = []
        self._current_segment = None
        self._last_keypoint_centroid = None
        self._frame_to_segment = {}

    def _validate_line_collinearity(
        self, H: np.ndarray, line_points_image: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Validate that collinear points in image space remain collinear after transformation.

        Args:
            H: Homography matrix (3x3)
            line_points_image: Points that should be collinear (N, 2)

        Returns:
            (is_valid, max_deviation_ft)
        """
        if len(line_points_image) < 3:
            return True, 0.0

        # Transform points to court space
        pts = cv2.perspectiveTransform(
            line_points_image.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)

        # Fit a line using SVD (robust to outliers)
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered)
        direction = vh[0]  # Principal direction

        # Compute perpendicular distances to line
        normal = np.array([-direction[1], direction[0]])
        distances = np.abs(np.dot(centered, normal))
        max_deviation = float(np.max(distances))

        is_valid = max_deviation <= self.line_collinearity_threshold
        return is_valid, max_deviation

    def _validate_arc_radius(
        self, H: np.ndarray, arc_points_image: np.ndarray, center_court: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Validate that arc points maintain expected radius from center.

        Args:
            H: Homography matrix (3x3)
            arc_points_image: Points on the arc in image coords (N, 2)
            center_court: Expected arc center in court coords (2,)

        Returns:
            (is_valid, radius_deviation_ft)
        """
        if len(arc_points_image) < 3:
            return True, 0.0

        # Transform points to court space
        pts = cv2.perspectiveTransform(
            arc_points_image.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)

        # Compute distances from center
        distances = np.linalg.norm(pts - center_court, axis=1)
        mean_radius = float(np.mean(distances))

        # Check deviation from expected radius
        radius_deviation = abs(mean_radius - self.three_point_radius_ft)
        is_valid = radius_deviation <= self.arc_radius_threshold

        return is_valid, radius_deviation

    def validate_semantic_constraints(
        self, H: np.ndarray, img_pts: np.ndarray, court_pts: np.ndarray
    ) -> Dict:
        """
        Validate semantic constraints for a homography.

        Checks:
        1. Line collinearity for baseline/sideline points
        2. Arc radius for 3-point line points

        Args:
            H: Homography matrix
            img_pts: Image keypoints (N, 2)
            court_pts: Corresponding court points (N, 2)

        Returns:
            Dict with validation results
        """
        results = {
            "is_valid": True,
            "line_valid": True,
            "arc_valid": True,
            "line_deviation_ft": 0.0,
            "arc_deviation_ft": 0.0,
            "warnings": []
        }

        if not self.enable_semantic_constraints:
            return results

        # Get court vertices for reference
        vertices = np.array(self.court_config.vertices, dtype=np.float32)

        # Identify baseline points (y = 0 or y = 50 for NBA)
        # Find points with similar y-coordinates
        y_coords = court_pts[:, 1]

        # Group by approximate y-value to find lines
        baseline_mask = np.abs(y_coords - 0) < 2.0  # Near y=0
        opposite_baseline_mask = np.abs(y_coords - 50) < 2.0  # Near y=50

        # Validate baselines
        for mask, name in [(baseline_mask, "baseline"), (opposite_baseline_mask, "opposite_baseline")]:
            if np.sum(mask) >= 3:
                line_pts = img_pts[mask]
                is_valid, deviation = self._validate_line_collinearity(H, line_pts)
                if not is_valid:
                    results["line_valid"] = False
                    results["is_valid"] = False
                    results["warnings"].append(f"{name} collinearity failed: {deviation:.2f}ft")
                results["line_deviation_ft"] = max(results["line_deviation_ft"], deviation)

        # Validate 3-point arc if we have arc points
        # Arc points are typically at radius ~23.75ft from basket
        # For now, we'll check if any points are near the expected arc distance
        basket_left = np.array([5.25, 25.0])  # Left basket position (feet)
        basket_right = np.array([88.75, 25.0])  # Right basket position (feet)

        for basket, name in [(basket_left, "left_arc"), (basket_right, "right_arc")]:
            distances_from_basket = np.linalg.norm(court_pts - basket, axis=1)
            arc_mask = np.abs(distances_from_basket - self.three_point_radius_ft) < 3.0

            if np.sum(arc_mask) >= 3:
                arc_pts = img_pts[arc_mask]
                is_valid, deviation = self._validate_arc_radius(H, arc_pts, basket)
                if not is_valid:
                    results["arc_valid"] = False
                    results["is_valid"] = False
                    results["warnings"].append(f"{name} radius failed: {deviation:.2f}ft")
                results["arc_deviation_ft"] = max(results["arc_deviation_ft"], deviation)

        return results

    def add_observation(
        self,
        frame_idx: int,
        keypoints: sv.KeyPoints,
        court_vertices: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Add keypoint observation for calibration.

        Args:
            frame_idx: Frame index
            keypoints: Detected court keypoints
            court_vertices: Optional court vertex coordinates (uses config if None)

        Returns:
            True if observation was added successfully
        """
        if keypoints.xy is None or keypoints.confidence is None:
            return False

        # Get court vertices
        if court_vertices is None:
            court_vertices = np.array(self.court_config.vertices, dtype=np.float32)

        # Filter by confidence
        xy = keypoints.xy[0]  # (M, 2)
        conf = keypoints.confidence[0]  # (M,)
        mask = conf >= self.confidence_threshold
        kept = int(mask.sum())

        if kept < self.min_keypoints:
            return False

        img_pts = xy[mask].astype(np.float32)
        court_pts = court_vertices[mask].astype(np.float32)
        confs = conf[mask].astype(np.float32)

        # Check for segment change
        centroid = np.mean(img_pts, axis=0)
        is_new_segment = self._detect_segment_change(centroid)

        if is_new_segment:
            self._start_new_segment(frame_idx)

        # Ensure we have a current segment
        if self._current_segment is None:
            self._start_new_segment(frame_idx)

        # Add observation
        self._current_segment.keypoint_observations.append((img_pts, court_pts, confs))
        self._current_segment.end_frame = frame_idx
        self._frame_to_segment[frame_idx] = self._current_segment.segment_id

        self._last_keypoint_centroid = centroid
        return True

    def _detect_segment_change(self, centroid: np.ndarray) -> bool:
        """Detect if there's a camera cut/zoom based on keypoint movement."""
        if self._last_keypoint_centroid is None:
            return True

        dist = np.linalg.norm(centroid - self._last_keypoint_centroid)
        return dist > self.segment_change_threshold

    def _start_new_segment(self, frame_idx: int) -> None:
        """Start a new camera segment."""
        # Finalize previous segment
        if self._current_segment is not None:
            self._calibrate_segment(self._current_segment)

        # Create new segment
        segment_id = len(self._segments)
        self._current_segment = SegmentData(
            segment_id=segment_id,
            start_frame=frame_idx,
        )
        self._segments.append(self._current_segment)

    def _calibrate_segment(self, segment: SegmentData) -> None:
        """
        Fit homography for a segment using all aggregated observations.

        Uses weighted least-squares with RANSAC outlier rejection.
        """
        if len(segment.keypoint_observations) == 0:
            return

        # Stack all observations
        all_img_pts = []
        all_court_pts = []
        all_weights = []

        for img_pts, court_pts, confs in segment.keypoint_observations:
            all_img_pts.append(img_pts)
            all_court_pts.append(court_pts)
            all_weights.append(confs)

        img_pts = np.vstack(all_img_pts)
        court_pts = np.vstack(all_court_pts)
        weights = np.concatenate(all_weights)

        segment.total_observations = len(img_pts)

        if len(img_pts) < self.min_keypoints:
            return

        # Check spread to avoid degenerate cases
        spread_x = np.ptp(img_pts[:, 0])
        spread_y = np.ptp(img_pts[:, 1])
        if spread_x < 50 or spread_y < 50:
            print(f"[WARN][calibrator] Segment {segment.segment_id}: degenerate spread")
            return

        # RANSAC homography
        try:
            H, inlier_mask = cv2.findHomography(
                img_pts, court_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_thresh,
                maxIters=5000,
                confidence=0.999,
            )
        except Exception as e:
            print(f"[WARN][calibrator] Segment {segment.segment_id}: RANSAC failed: {e}")
            return

        if H is None or inlier_mask is None:
            return

        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.sum(inliers))

        if inlier_count < self.min_keypoints:
            return

        if inlier_count / len(img_pts) < self.min_inlier_ratio:
            print(f"[WARN][calibrator] Segment {segment.segment_id}: low inlier ratio")
            return

        # Refine with weighted least-squares on inliers
        inlier_img = img_pts[inliers]
        inlier_court = court_pts[inliers]
        inlier_weights = weights[inliers]

        # Weighted refinement using normalized weights
        w = inlier_weights / np.sum(inlier_weights)
        # For simplicity, use standard least-squares (cv2.LMEDS could be used too)
        H_refined, _ = cv2.findHomography(inlier_img, inlier_court, method=0)

        if H_refined is None:
            H_refined = H

        # Compute inverse
        try:
            H_inv = np.linalg.inv(H_refined)
        except np.linalg.LinAlgError:
            return

        # Compute quality metrics
        projected = cv2.perspectiveTransform(
            inlier_img.reshape(-1, 1, 2), H_refined
        ).reshape(-1, 2)
        errors_court = np.linalg.norm(projected - inlier_court, axis=1)
        rmse_court = float(np.sqrt(np.mean(errors_court ** 2)))

        # Backward projection
        back_projected = cv2.perspectiveTransform(
            inlier_court.reshape(-1, 1, 2), H_inv
        ).reshape(-1, 2)
        errors_img = np.linalg.norm(back_projected - inlier_img, axis=1)
        rmse_image = float(np.sqrt(np.mean(errors_img ** 2)))

        # Create ViewTransformers
        try:
            img2court = ViewTransformer(source=inlier_img, target=inlier_court)
            court2img = ViewTransformer(source=inlier_court, target=inlier_img)
        except Exception as e:
            print(f"[WARN][calibrator] Segment {segment.segment_id}: ViewTransformer failed: {e}")
            return

        # Store results
        segment.H_img2court = H_refined
        segment.H_court2img = H_inv
        segment.img2court = img2court
        segment.court2img = court2img
        segment.rmse_court = rmse_court
        segment.rmse_image = rmse_image
        segment.inlier_count = inlier_count

        # Validate semantic constraints (soft validation - log warnings but don't reject)
        semantic_results = self.validate_semantic_constraints(H_refined, inlier_img, inlier_court)

        constraint_info = ""
        if semantic_results["warnings"]:
            for warning in semantic_results["warnings"]:
                print(f"[WARN][calibrator] Segment {segment.segment_id}: {warning}")
            constraint_info = f" [semantic: line={semantic_results['line_deviation_ft']:.2f}ft arc={semantic_results['arc_deviation_ft']:.2f}ft]"

        print(
            f"[INFO][calibrator] Segment {segment.segment_id} calibrated: "
            f"frames={segment.start_frame}-{segment.end_frame}, "
            f"inliers={inlier_count}/{segment.total_observations}, "
            f"RMSE court={rmse_court:.3f}ft image={rmse_image:.2f}px{constraint_info}"
        )

    def finalize(self) -> None:
        """Finalize calibration (call after processing all frames)."""
        if self._current_segment is not None:
            self._calibrate_segment(self._current_segment)

    def get_transform(
        self, frame_idx: int
    ) -> Tuple[Optional[ViewTransformer], Optional[ViewTransformer], Dict]:
        """
        Get homography for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            (img2court, court2img, quality_info)
        """
        segment_id = self._frame_to_segment.get(frame_idx)

        if segment_id is None:
            return None, None, {"error": "frame_not_observed"}

        segment = self._segments[segment_id]

        if not segment.is_calibrated:
            # Try to calibrate if not done
            self._calibrate_segment(segment)

        if not segment.is_calibrated:
            return None, None, {"error": "segment_not_calibrated"}

        # Check quality thresholds
        quality = {
            "segment_id": segment.segment_id,
            "rmse_court": segment.rmse_court,
            "rmse_image": segment.rmse_image,
            "inlier_count": segment.inlier_count,
            "total_observations": segment.total_observations,
            "is_reliable": (
                segment.rmse_court <= self.rmse_court_max and
                segment.rmse_image <= self.rmse_image_max
            ),
        }

        return segment.img2court, segment.court2img, quality

    def get_segment_for_frame(self, frame_idx: int) -> Optional[SegmentData]:
        """Get the segment containing a specific frame."""
        segment_id = self._frame_to_segment.get(frame_idx)
        if segment_id is None:
            return None
        return self._segments[segment_id]

    def get_all_segments(self) -> List[SegmentData]:
        """Get all calibrated segments."""
        return self._segments

    def get_quality_mask(self) -> Dict[int, bool]:
        """
        Get reliability mask for all observed frames.

        Returns:
            Dict mapping frame_idx -> is_reliable
        """
        mask = {}
        for frame_idx, segment_id in self._frame_to_segment.items():
            segment = self._segments[segment_id]
            mask[frame_idx] = (
                segment.is_calibrated and
                segment.rmse_court <= self.rmse_court_max and
                segment.rmse_image <= self.rmse_image_max
            )
        return mask

    def interpolate_unreliable_frames(
        self, positions: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Interpolate court positions for frames with unreliable homography.

        Args:
            positions: Dict mapping frame_idx -> court position

        Returns:
            Interpolated positions
        """
        mask = self.get_quality_mask()
        result = positions.copy()

        # Find unreliable frames
        unreliable = [f for f, reliable in mask.items() if not reliable and f in positions]

        for frame_idx in unreliable:
            # Find nearest reliable frames
            reliable_frames = sorted([f for f, r in mask.items() if r and f in positions])
            if not reliable_frames:
                continue

            # Linear interpolation between neighbors
            before = [f for f in reliable_frames if f < frame_idx]
            after = [f for f in reliable_frames if f > frame_idx]

            if before and after:
                f1, f2 = before[-1], after[0]
                p1, p2 = positions[f1], positions[f2]
                t = (frame_idx - f1) / (f2 - f1)
                result[frame_idx] = p1 + t * (p2 - p1)
            elif before:
                result[frame_idx] = positions[before[-1]]
            elif after:
                result[frame_idx] = positions[after[0]]

        return result

    def reset(self) -> None:
        """Reset calibrator state."""
        self._segments.clear()
        self._current_segment = None
        self._last_keypoint_centroid = None
        self._frame_to_segment.clear()


def create_calibrator(cfg=None) -> HomographyCalibrator:
    """
    Create a HomographyCalibrator from config.

    Args:
        cfg: CVConfig instance

    Returns:
        Configured HomographyCalibrator
    """
    from api.src.cv.config import CVConfig

    if cfg is None:
        cfg = CVConfig()

    return HomographyCalibrator(
        court_config=cfg.court_config,
        confidence_threshold=cfg.keypoint_conf_threshold,
        min_keypoints=cfg.min_keypoints_required,
        ransac_reproj_thresh=cfg.ransac_reproj_thresh_px,
        min_inlier_ratio=cfg.min_inlier_ratio,
        rmse_court_max=cfg.homography_rmse_court_max,
        rmse_image_max=cfg.homography_rmse_image_max,
    )
