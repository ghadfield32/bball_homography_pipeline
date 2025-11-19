# api/src/cv/shot_arc.py
"""
Shot arc analysis module for basketball trajectory metrics.

Tracks ball trajectory during shots and computes:
- Release angle and velocity
- Arc height (apex)
- Entry angle
- Trajectory smoothing and validation

Uses ball detection model to track ball positions frame-by-frame,
then fits parabolic curves for physics-based analysis.

Usage:
    arc_analyzer = ShotArcAnalyzer(cfg)

    for frame_idx, frame in enumerate(video):
        ball_det = arc_analyzer.detect_ball(frame)
        arc_analyzer.add_observation(frame_idx, ball_det)

    # After shot detected
    metrics = arc_analyzer.analyze_shot_arc(start_frame, end_frame)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from sports import ViewTransformer


@dataclass
class BallObservation:
    """Single ball detection observation."""
    frame_idx: int
    position_image: np.ndarray  # [x, y] in pixels
    position_court: Optional[np.ndarray] = None  # [x, y] in feet
    confidence: float = 1.0
    bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2]


@dataclass
class ArcMetrics:
    """Shot arc analysis metrics."""
    # Trajectory
    trajectory_points: List[Tuple[float, float]]  # [(x, y), ...] in image coords
    trajectory_court: Optional[List[Tuple[float, float]]] = None  # court coords

    # Angles (degrees)
    release_angle: float = 0.0  # Angle at release point
    entry_angle: float = 0.0  # Angle entering basket
    arc_angle: float = 0.0  # Peak angle of arc

    # Heights
    apex_height_px: float = 0.0  # Highest point in pixels
    release_height_px: float = 0.0  # Release point height
    apex_height_ft: Optional[float] = None  # If court coords available

    # Velocity (pixels/frame or ft/second)
    release_velocity_px: float = 0.0
    release_velocity_fps: Optional[float] = None  # feet per second

    # Quality metrics
    trajectory_length: int = 0
    fit_r_squared: float = 0.0
    is_valid: bool = False


@dataclass
class ShotArcAnalyzer:
    """
    Shot arc analyzer for ball trajectory metrics.

    Tracks ball through shot sequence and computes physics-based metrics.
    """
    # Ball detection parameters
    confidence_threshold: float = 0.3
    min_trajectory_points: int = 5
    velocity_window_frames: int = 3
    angle_smoothing: int = 3

    # Court dimensions for conversions
    rim_height_ft: float = 10.0
    ball_diameter_ft: float = 0.78  # ~9.4 inches

    # Internal state
    _observations: List[BallObservation] = field(default_factory=list)
    _ball_model = None
    _fps: float = 30.0

    def __post_init__(self):
        """Initialize state."""
        self._observations = []

    def load_model(self, model_id: str = "basketball-detection/1") -> bool:
        """
        Load ball detection model.

        Args:
            model_id: Roboflow model ID for ball detection

        Returns:
            True if model loaded successfully
        """
        try:
            import os
            from inference import get_model

            api_key = os.getenv("ROBOFLOW_API_KEY") or os.getenv("INFERENCE_API_KEY")
            if not api_key:
                print("[WARN][shot_arc] No API key for ball model")
                return False

            self._ball_model = get_model(model_id=model_id, api_key=api_key)
            print(f"[INFO][shot_arc] Loaded ball model: {model_id}")
            return True

        except Exception as e:
            print(f"[WARN][shot_arc] Failed to load ball model: {e}")
            return False

    def detect_ball(self, frame: np.ndarray) -> Optional[BallObservation]:
        """
        Detect ball in frame.

        Args:
            frame: BGR image

        Returns:
            BallObservation or None if no ball detected
        """
        if self._ball_model is None:
            return None

        try:
            result = self._ball_model.infer(frame)[0]

            if not hasattr(result, "predictions") or not result.predictions:
                return None

            # Get highest confidence ball detection
            best_pred = None
            best_conf = 0.0

            for pred in result.predictions:
                if pred.confidence > best_conf and pred.confidence >= self.confidence_threshold:
                    best_pred = pred
                    best_conf = pred.confidence

            if best_pred is None:
                return None

            # Extract center position
            cx = best_pred.x
            cy = best_pred.y
            w = best_pred.width
            h = best_pred.height

            bbox = np.array([
                cx - w / 2, cy - h / 2,
                cx + w / 2, cy + h / 2
            ])

            return BallObservation(
                frame_idx=-1,  # Set by caller
                position_image=np.array([cx, cy]),
                confidence=best_conf,
                bbox=bbox,
            )

        except Exception as e:
            print(f"[WARN][shot_arc] Ball detection failed: {e}")
            return None

    def add_observation(
        self,
        frame_idx: int,
        observation: Optional[BallObservation],
        img2court: Optional[ViewTransformer] = None,
    ) -> None:
        """
        Add ball observation for a frame.

        Args:
            frame_idx: Frame number
            observation: Ball detection or None
            img2court: Optional transform to court coordinates
        """
        if observation is None:
            return

        observation.frame_idx = frame_idx

        # Transform to court coords if available
        if img2court is not None:
            try:
                pt = observation.position_image.reshape(1, 2)
                court_pt = img2court.transform_points(pt)
                observation.position_court = court_pt[0]
            except Exception:
                pass

        self._observations.append(observation)

    def get_trajectory_segment(
        self,
        start_frame: int,
        end_frame: int,
    ) -> List[BallObservation]:
        """Get observations within a frame range."""
        return [
            obs for obs in self._observations
            if start_frame <= obs.frame_idx <= end_frame
        ]

    def analyze_shot_arc(
        self,
        start_frame: int,
        end_frame: int,
        fps: float = 30.0,
    ) -> ArcMetrics:
        """
        Analyze shot arc for a shot sequence.

        Args:
            start_frame: Shot start frame
            end_frame: Shot end frame
            fps: Video frame rate

        Returns:
            ArcMetrics with trajectory analysis
        """
        self._fps = fps
        segment = self.get_trajectory_segment(start_frame, end_frame)

        # Initialize metrics
        metrics = ArcMetrics(
            trajectory_points=[],
            trajectory_length=len(segment),
            is_valid=False,
        )

        if len(segment) < self.min_trajectory_points:
            return metrics

        # Extract trajectory points
        frames = np.array([obs.frame_idx for obs in segment])
        points = np.array([obs.position_image for obs in segment])

        # Smooth trajectory
        if len(points) > self.angle_smoothing:
            points = self._smooth_trajectory(points, self.angle_smoothing)

        # Store trajectory
        metrics.trajectory_points = [(float(p[0]), float(p[1])) for p in points]

        # Fit parabola to trajectory
        fit_result = self._fit_parabola(frames, points)
        if fit_result is None:
            return metrics

        coeffs, r_squared = fit_result
        metrics.fit_r_squared = r_squared

        # Compute angles
        metrics.release_angle = self._compute_release_angle(points, frames)
        metrics.entry_angle = self._compute_entry_angle(points, frames)

        # Find apex (highest point)
        apex_idx = np.argmin(points[:, 1])  # Min y = highest in image coords
        metrics.apex_height_px = float(points[apex_idx, 1])
        metrics.release_height_px = float(points[0, 1])

        # Compute velocity
        metrics.release_velocity_px = self._compute_velocity(points, frames, 0)

        # Court coordinates if available
        court_points = [obs.position_court for obs in segment if obs.position_court is not None]
        if court_points:
            metrics.trajectory_court = [(float(p[0]), float(p[1])) for p in court_points]

            # Estimate height in feet (rough approximation)
            # Assumes camera is roughly level with court
            if len(court_points) >= 2:
                pixel_per_foot = self._estimate_scale(segment)
                if pixel_per_foot > 0:
                    height_diff_px = metrics.release_height_px - metrics.apex_height_px
                    metrics.apex_height_ft = height_diff_px / pixel_per_foot + self.rim_height_ft
                    metrics.release_velocity_fps = metrics.release_velocity_px / pixel_per_foot * fps

        metrics.is_valid = r_squared > 0.7 and len(points) >= self.min_trajectory_points

        return metrics

    def _smooth_trajectory(
        self,
        points: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Apply moving average smoothing to trajectory."""
        if len(points) <= window:
            return points

        smoothed = np.copy(points).astype(float)

        for i in range(len(points)):
            start = max(0, i - window // 2)
            end = min(len(points), i + window // 2 + 1)
            smoothed[i] = np.mean(points[start:end], axis=0)

        return smoothed

    def _fit_parabola(
        self,
        frames: np.ndarray,
        points: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Fit parabola to trajectory (y = ax^2 + bx + c).

        Returns:
            (coefficients, r_squared) or None if fit fails
        """
        try:
            # Normalize frames to start at 0
            t = frames - frames[0]
            y = points[:, 1]  # Vertical position

            # Fit quadratic
            coeffs = np.polyfit(t, y, 2)
            y_pred = np.polyval(coeffs, t)

            # Compute R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return coeffs, r_squared

        except Exception:
            return None

    def _compute_release_angle(
        self,
        points: np.ndarray,
        frames: np.ndarray,
    ) -> float:
        """Compute release angle in degrees."""
        if len(points) < 2:
            return 0.0

        # Use first few points to estimate release angle
        n = min(self.velocity_window_frames, len(points))
        dx = points[n - 1, 0] - points[0, 0]
        dy = points[n - 1, 1] - points[0, 1]

        # Angle relative to horizontal (positive = upward)
        # In image coords, y increases downward, so negate
        angle_rad = np.arctan2(-dy, dx)
        return float(np.degrees(angle_rad))

    def _compute_entry_angle(
        self,
        points: np.ndarray,
        frames: np.ndarray,
    ) -> float:
        """Compute entry angle in degrees."""
        if len(points) < 2:
            return 0.0

        # Use last few points to estimate entry angle
        n = min(self.velocity_window_frames, len(points))
        dx = points[-1, 0] - points[-n, 0]
        dy = points[-1, 1] - points[-n, 1]

        # Entry angle (typically negative = downward into basket)
        angle_rad = np.arctan2(-dy, dx)
        return float(np.degrees(angle_rad))

    def _compute_velocity(
        self,
        points: np.ndarray,
        frames: np.ndarray,
        idx: int,
    ) -> float:
        """Compute velocity at a point (pixels/frame)."""
        if len(points) < 2:
            return 0.0

        n = min(self.velocity_window_frames, len(points) - idx)
        if n < 2:
            return 0.0

        # Distance over frames
        dist = np.linalg.norm(points[idx + n - 1] - points[idx])
        frame_diff = frames[idx + n - 1] - frames[idx]

        return float(dist / frame_diff) if frame_diff > 0 else 0.0

    def _estimate_scale(
        self,
        segment: List[BallObservation],
    ) -> float:
        """
        Estimate pixels per foot from ball size.

        Uses ball diameter as reference.
        """
        # Get average ball bbox size
        sizes = []
        for obs in segment:
            if obs.bbox is not None:
                w = obs.bbox[2] - obs.bbox[0]
                h = obs.bbox[3] - obs.bbox[1]
                sizes.append((w + h) / 2)

        if not sizes:
            return 0.0

        avg_size_px = np.mean(sizes)

        # Ball diameter is ~9.4 inches = 0.78 feet
        return avg_size_px / self.ball_diameter_ft

    def get_all_trajectories(self) -> List[List[Tuple[float, float]]]:
        """Get all recorded trajectory points grouped by continuity."""
        if not self._observations:
            return []

        trajectories = []
        current = []
        prev_frame = -999

        for obs in sorted(self._observations, key=lambda o: o.frame_idx):
            if obs.frame_idx - prev_frame > 5:  # Gap in observations
                if current:
                    trajectories.append(current)
                current = []

            current.append((float(obs.position_image[0]), float(obs.position_image[1])))
            prev_frame = obs.frame_idx

        if current:
            trajectories.append(current)

        return trajectories

    def reset(self) -> None:
        """Reset all observations."""
        self._observations.clear()


def create_shot_arc_analyzer(cfg=None) -> ShotArcAnalyzer:
    """
    Create ShotArcAnalyzer from config.

    Args:
        cfg: CVConfig instance

    Returns:
        Configured ShotArcAnalyzer
    """
    if cfg is None:
        return ShotArcAnalyzer()

    analyzer = ShotArcAnalyzer(
        confidence_threshold=getattr(cfg, "ball_confidence_threshold", 0.3),
        min_trajectory_points=getattr(cfg, "arc_min_trajectory_points", 5),
        velocity_window_frames=getattr(cfg, "arc_velocity_window_frames", 3),
        angle_smoothing=getattr(cfg, "arc_angle_smoothing", 3),
    )

    # Load ball model if arc analysis enabled
    if getattr(cfg, "enable_shot_arc_analysis", False):
        ball_model_id = getattr(cfg, "ball_model_id", "basketball-detection/1")
        analyzer.load_model(ball_model_id)

    return analyzer
