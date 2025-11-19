# api/src/cv/tracker.py
"""
Multi-object tracking module for basketball player tracking.

Uses supervision's ByteTrack for robust tracking-by-detection with:
- Persistent track IDs across frames
- Track history in both image and court coordinates
- Team label attachment to tracks (stable across frames)
- Speed, distance, and movement analytics

Integration: Insert after detect_players() and before shot tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import supervision as sv
from sports import ViewTransformer


@dataclass
class TrackState:
    """State for a single tracked player across frames."""
    track_id: int
    team_id: int = -1  # -1 = unknown, 0 = team A, 1 = team B, 2 = referee

    # Rolling history (most recent last)
    history: List[Dict] = field(default_factory=list)

    # Team label voting (for stability)
    team_votes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Derived analytics
    total_distance_court_ft: float = 0.0
    max_speed_fps: float = 0.0  # feet per second

    def add_observation(
        self,
        frame_idx: int,
        bbox_image: np.ndarray,
        point_image: np.ndarray,
        point_court: Optional[np.ndarray],
        team_id: int,
        confidence: float,
        fps: float = 30.0,
    ) -> None:
        """
        Add a new observation for this track.

        Args:
            frame_idx: Current frame number
            bbox_image: [x1, y1, x2, y2] bounding box in image coords
            point_image: [x, y] bottom-center in image coords
            point_court: [x, y] in court coords (feet), or None if H unavailable
            team_id: Team classification for this frame
            confidence: Detection confidence
            fps: Video frame rate for speed calculation
        """
        obs = {
            "frame_idx": frame_idx,
            "bbox_image": bbox_image.copy() if bbox_image is not None else None,
            "point_image": point_image.copy() if point_image is not None else None,
            "point_court": point_court.copy() if point_court is not None else None,
            "team_id": team_id,
            "confidence": confidence,
        }

        # Update distance and speed if we have court coords
        if point_court is not None and len(self.history) > 0:
            prev = self.history[-1]
            if prev["point_court"] is not None:
                prev_pt = np.array(prev["point_court"])
                curr_pt = np.array(point_court)
                dist = float(np.linalg.norm(curr_pt - prev_pt))
                self.total_distance_court_ft += dist

                # Speed in feet per second
                frame_diff = frame_idx - prev["frame_idx"]
                if frame_diff > 0:
                    time_diff = frame_diff / fps
                    speed = dist / time_diff
                    self.max_speed_fps = max(self.max_speed_fps, speed)

        self.history.append(obs)

        # Vote for team (for stability)
        if team_id >= 0:
            self.team_votes[team_id] += 1
            # Update team_id to majority vote
            self.team_id = max(self.team_votes, key=self.team_votes.get)

    def get_recent_positions(self, n: int = 10) -> np.ndarray:
        """Get last n court positions as (n, 2) array."""
        pts = []
        for obs in reversed(self.history[-n:]):
            if obs["point_court"] is not None:
                pts.append(obs["point_court"])
        if not pts:
            return np.empty((0, 2), dtype=float)
        return np.array(pts[::-1], dtype=float)

    @property
    def is_active(self) -> bool:
        """Track is active if it has recent observations."""
        return len(self.history) > 0

    @property
    def last_frame(self) -> int:
        """Last frame this track was observed."""
        return self.history[-1]["frame_idx"] if self.history else -1


@dataclass
class PlayerTracker:
    """
    Multi-object tracker for basketball players using ByteTrack.

    Maintains persistent track IDs and histories across frames.
    Integrates with team classification to stabilize labels.

    Usage:
        tracker = PlayerTracker()
        for frame_idx, frame in enumerate(video):
            detections = detect_players(frame, model, cfg)
            tracked = tracker.update(detections, frame_idx, img2court)
            # tracked.tracker_id now has persistent IDs
    """
    # ByteTrack parameters
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30
    minimum_consecutive_frames: int = 1

    # Track state
    _tracker: sv.ByteTrack = field(init=False)
    _tracks: Dict[int, TrackState] = field(default_factory=dict)
    _frame_count: int = field(default=0, init=False)

    def __post_init__(self):
        """Initialize ByteTrack tracker."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=self.frame_rate,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )
        self._tracks = {}
        self._frame_count = 0

    def update(
        self,
        detections: sv.Detections,
        frame_idx: int,
        img2court: Optional[ViewTransformer] = None,
        team_assignments: Optional[np.ndarray] = None,
    ) -> sv.Detections:
        """
        Update tracker with new detections.

        Args:
            detections: sv.Detections from player detection model
            frame_idx: Current frame number
            img2court: Optional ViewTransformer to convert to court coords
            team_assignments: Optional array of team IDs per detection (0=A, 1=B, 2=ref)

        Returns:
            sv.Detections with tracker_id field populated
        """
        self._frame_count = frame_idx

        if len(detections) == 0:
            return detections

        # Run ByteTrack association
        tracked = self._tracker.update_with_detections(detections)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return tracked

        # Get bottom-center anchors (player feet position)
        anchors = tracked.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        # Transform to court if we have homography
        court_pts = None
        if img2court is not None and anchors is not None and len(anchors) > 0:
            try:
                court_pts = img2court.transform_points(anchors)
            except Exception as e:
                print(f"[WARN][tracker] court transform failed: {e}")
                court_pts = None

        # Update track states
        for i, track_id in enumerate(tracked.tracker_id):
            track_id = int(track_id)

            # Create new track if needed
            if track_id not in self._tracks:
                self._tracks[track_id] = TrackState(track_id=track_id)

            # Get team assignment
            team_id = -1
            if team_assignments is not None and i < len(team_assignments):
                team_id = int(team_assignments[i])

            # Get positions
            bbox = tracked.xyxy[i] if tracked.xyxy is not None else None
            pt_img = anchors[i] if anchors is not None else None
            pt_court = court_pts[i] if court_pts is not None else None
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0

            # Add observation
            self._tracks[track_id].add_observation(
                frame_idx=frame_idx,
                bbox_image=bbox,
                point_image=pt_img,
                point_court=pt_court,
                team_id=team_id,
                confidence=conf,
                fps=self.frame_rate,
            )

        return tracked

    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Get track state by ID."""
        return self._tracks.get(track_id)

    def get_active_tracks(self, max_age: int = 30) -> List[TrackState]:
        """Get tracks active within last max_age frames."""
        return [
            t for t in self._tracks.values()
            if self._frame_count - t.last_frame <= max_age
        ]

    def get_team_tracks(self, team_id: int) -> List[TrackState]:
        """Get all tracks for a specific team."""
        return [t for t in self._tracks.values() if t.team_id == team_id]

    def get_track_positions(
        self, frame_idx: int, as_court: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Get positions of all tracks at a specific frame.

        Args:
            frame_idx: Frame to query
            as_court: If True, return court coords; else image coords

        Returns:
            Dict mapping track_id -> [x, y] position
        """
        positions = {}
        key = "point_court" if as_court else "point_image"

        for track_id, track in self._tracks.items():
            for obs in track.history:
                if obs["frame_idx"] == frame_idx and obs[key] is not None:
                    positions[track_id] = obs[key]
                    break

        return positions

    def get_analytics(self) -> Dict:
        """Get aggregate tracking analytics."""
        active = self.get_active_tracks()

        if not active:
            return {
                "total_tracks": len(self._tracks),
                "active_tracks": 0,
                "team_counts": {0: 0, 1: 0, 2: 0},
                "avg_distance_ft": 0.0,
                "max_speed_fps": 0.0,
            }

        team_counts = defaultdict(int)
        distances = []
        speeds = []

        for track in active:
            team_counts[track.team_id] += 1
            distances.append(track.total_distance_court_ft)
            speeds.append(track.max_speed_fps)

        return {
            "total_tracks": len(self._tracks),
            "active_tracks": len(active),
            "team_counts": dict(team_counts),
            "avg_distance_ft": float(np.mean(distances)) if distances else 0.0,
            "max_speed_fps": float(np.max(speeds)) if speeds else 0.0,
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracker.reset()
        self._tracks.clear()
        self._frame_count = 0


def assign_teams_to_detections(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    groups: Dict[str, Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Map team assignments back to detections array.

    Args:
        frame_bgr: Current frame
        detections: Detection results
        groups: Output from group_players_into_teams()

    Returns:
        Array of team IDs (0=A, 1=B, 2=ref) matching detection order
    """
    n = len(detections)
    assignments = np.full(n, -1, dtype=int)

    for team_id, (key, group) in enumerate([("A", 0), ("B", 1), ("REF", 2)]):
        idx = groups[key]["idx"]
        for i in idx:
            if i < n:
                assignments[i] = team_id

    return assignments


# Convenience function to integrate with existing pipeline
def create_tracker(cfg=None) -> PlayerTracker:
    """
    Create a PlayerTracker with config-based parameters.

    Args:
        cfg: CVConfig instance (optional)

    Returns:
        Configured PlayerTracker
    """
    if cfg is None:
        return PlayerTracker()

    # Extract relevant config if available
    fps = getattr(cfg, "video_fps", 30)

    return PlayerTracker(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=fps,
        minimum_consecutive_frames=1,
    )
