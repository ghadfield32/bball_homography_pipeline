# api/src/cv/pose_pipeline.py
"""
Pose estimation pipeline for basketball player biomechanics analysis.

Uses YOLO-pose models to extract 17-keypoint skeletons for players,
then transforms to court coordinates for biomechanics analysis.

Key features:
- Per-player pose estimation from crops or full-frame detection
- Joint trajectories in court coordinates
- Shot-specific metrics (release point, joint sequences)
- Integration with player tracking

COCO 17-keypoint format:
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

import numpy as np
from sports import ViewTransformer


class JointID(IntEnum):
    """COCO 17-keypoint joint indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (JointID.LEFT_ANKLE, JointID.LEFT_KNEE),
    (JointID.LEFT_KNEE, JointID.LEFT_HIP),
    (JointID.RIGHT_ANKLE, JointID.RIGHT_KNEE),
    (JointID.RIGHT_KNEE, JointID.RIGHT_HIP),
    (JointID.LEFT_HIP, JointID.RIGHT_HIP),
    (JointID.LEFT_SHOULDER, JointID.LEFT_HIP),
    (JointID.RIGHT_SHOULDER, JointID.RIGHT_HIP),
    (JointID.LEFT_SHOULDER, JointID.RIGHT_SHOULDER),
    (JointID.LEFT_SHOULDER, JointID.LEFT_ELBOW),
    (JointID.LEFT_ELBOW, JointID.LEFT_WRIST),
    (JointID.RIGHT_SHOULDER, JointID.RIGHT_ELBOW),
    (JointID.RIGHT_ELBOW, JointID.RIGHT_WRIST),
    (JointID.LEFT_SHOULDER, JointID.LEFT_EAR),
    (JointID.RIGHT_SHOULDER, JointID.RIGHT_EAR),
    (JointID.NOSE, JointID.LEFT_EYE),
    (JointID.NOSE, JointID.RIGHT_EYE),
    (JointID.LEFT_EYE, JointID.LEFT_EAR),
    (JointID.RIGHT_EYE, JointID.RIGHT_EAR),
]


@dataclass
class PoseObservation:
    """Single pose observation for a player."""
    frame_idx: int
    track_id: int

    # Keypoints in image coordinates (17, 2)
    keypoints_image: np.ndarray
    # Keypoint confidences (17,)
    confidences: np.ndarray
    # Keypoints in court coordinates (17, 2) - only x, y (z requires 3D reconstruction)
    keypoints_court: Optional[np.ndarray] = None

    @property
    def is_valid(self) -> bool:
        """Check if pose has enough confident keypoints."""
        return np.sum(self.confidences > 0.5) >= 8


@dataclass
class PlayerPoseHistory:
    """Pose history for a single tracked player."""
    track_id: int
    observations: List[PoseObservation] = field(default_factory=list)

    def add_observation(self, obs: PoseObservation) -> None:
        """Add pose observation."""
        self.observations.append(obs)

    def get_joint_trajectory(
        self,
        joint_id: JointID,
        as_court: bool = True,
        min_confidence: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory of a specific joint over time.

        Args:
            joint_id: Joint to extract
            as_court: If True, return court coords; else image
            min_confidence: Minimum confidence threshold

        Returns:
            (frames, positions) where positions is (N, 2)
        """
        frames = []
        positions = []

        for obs in self.observations:
            if obs.confidences[joint_id] < min_confidence:
                continue

            if as_court and obs.keypoints_court is not None:
                pos = obs.keypoints_court[joint_id]
            else:
                pos = obs.keypoints_image[joint_id]

            frames.append(obs.frame_idx)
            positions.append(pos)

        return (
            np.array(frames, dtype=int),
            np.array(positions, dtype=float) if positions else np.empty((0, 2))
        )

    def get_release_point_estimate(
        self,
        hand: str = "right",
        fps: float = 30.0,
    ) -> Optional[Dict]:
        """
        Estimate shot release point from wrist trajectory.

        The release point is approximately where:
        1. Wrist reaches highest point during shooting motion
        2. Vertical velocity changes from positive to negative

        Args:
            hand: "left" or "right"
            fps: Video frame rate

        Returns:
            Dict with release info or None if not detected
        """
        joint_id = JointID.RIGHT_WRIST if hand == "right" else JointID.LEFT_WRIST
        frames, positions = self.get_joint_trajectory(joint_id, as_court=False)

        if len(frames) < 5:
            return None

        # Find local maxima in y (image y increases downward, so we look for minima)
        y_coords = positions[:, 1]

        # Compute velocity
        dt = 1.0 / fps
        velocities = np.diff(y_coords) / dt

        # Find zero crossings (velocity sign change from negative to positive)
        # In image coords, going up = negative velocity
        release_candidates = []
        for i in range(1, len(velocities)):
            if velocities[i-1] < 0 and velocities[i] >= 0:  # Was going up, now going down
                release_candidates.append(i)

        if not release_candidates:
            return None

        # Take the last candidate as likely release
        release_idx = release_candidates[-1]

        return {
            "frame_idx": int(frames[release_idx]),
            "position_image": positions[release_idx].tolist(),
            "wrist_height_px": float(y_coords[release_idx]),
            "velocity_before": float(velocities[release_idx - 1]) if release_idx > 0 else 0.0,
            "hand": hand,
        }


@dataclass
class PosePipeline:
    """
    Pipeline for extracting and analyzing player poses.

    Uses YOLO-pose models (yolov8n-pose, yolov8s-pose, etc.) for
    skeleton detection, then transforms to court coordinates.

    Usage:
        pose_pipeline = PosePipeline()
        pose_pipeline.load_model("yolov8n-pose")

        for frame_idx, frame in enumerate(video):
            poses = pose_pipeline.detect_poses(frame)
            pose_pipeline.add_observations(frame_idx, poses, tracked_detections, img2court)
    """
    model_name: str = "yolov8n-pose"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.7

    # Internal state
    _model = None
    _pose_histories: Dict[int, PlayerPoseHistory] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize state."""
        self._pose_histories = {}

    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Load YOLO-pose model.

        Args:
            model_name: Model name (yolov8n-pose, yolov8s-pose, yolov8m-pose)
        """
        if model_name:
            self.model_name = model_name

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            print(f"[INFO][pose] Loaded {self.model_name}")
        except Exception as e:
            print(f"[ERROR][pose] Failed to load {self.model_name}: {e}")
            self._model = None

    def detect_poses(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect poses in a frame.

        Args:
            frame: BGR image

        Returns:
            List of pose detections with keypoints and boxes
        """
        if self._model is None:
            self.load_model()

        if self._model is None:
            return []

        try:
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            print(f"[WARN][pose] Inference failed: {e}")
            return []

        poses = []

        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else None

            for i in range(len(keypoints_data)):
                kp = keypoints_data[i]  # (17, 3) - x, y, conf
                xy = kp[:, :2]  # (17, 2)
                conf = kp[:, 2]  # (17,)

                pose = {
                    "keypoints_image": xy,
                    "confidences": conf,
                    "bbox": boxes[i] if boxes is not None else None,
                }
                poses.append(pose)

        return poses

    def match_poses_to_tracks(
        self,
        poses: List[Dict],
        tracked_detections,
    ) -> Dict[int, Dict]:
        """
        Match pose detections to tracked players by bbox overlap.

        Args:
            poses: Pose detections from detect_poses()
            tracked_detections: sv.Detections with tracker_id

        Returns:
            Dict mapping track_id -> pose dict
        """
        if tracked_detections.tracker_id is None or len(poses) == 0:
            return {}

        matched = {}

        for pose in poses:
            if pose["bbox"] is None:
                continue

            pose_box = pose["bbox"]  # [x1, y1, x2, y2]
            best_iou = 0.0
            best_track_id = None

            for i, track_id in enumerate(tracked_detections.tracker_id):
                track_box = tracked_detections.xyxy[i]

                # Compute IoU
                iou = self._compute_iou(pose_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = int(track_id)

            if best_track_id is not None and best_iou > 0.3:
                matched[best_track_id] = pose

        return matched

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def add_observations(
        self,
        frame_idx: int,
        poses: List[Dict],
        tracked_detections,
        img2court: Optional[ViewTransformer] = None,
    ) -> None:
        """
        Add pose observations for tracked players.

        Args:
            frame_idx: Current frame
            poses: Pose detections
            tracked_detections: sv.Detections with tracker_id
            img2court: Optional homography transform
        """
        matched = self.match_poses_to_tracks(poses, tracked_detections)

        for track_id, pose in matched.items():
            # Create pose history if needed
            if track_id not in self._pose_histories:
                self._pose_histories[track_id] = PlayerPoseHistory(track_id=track_id)

            # Transform keypoints to court
            keypoints_court = None
            if img2court is not None:
                try:
                    kp_image = pose["keypoints_image"]
                    # Only transform confident keypoints
                    mask = pose["confidences"] > 0.3
                    if np.any(mask):
                        valid_kp = kp_image[mask]
                        transformed = img2court.transform_points(valid_kp)
                        keypoints_court = np.full_like(kp_image, np.nan)
                        keypoints_court[mask] = transformed
                except Exception as e:
                    print(f"[WARN][pose] Court transform failed: {e}")

            # Create observation
            obs = PoseObservation(
                frame_idx=frame_idx,
                track_id=track_id,
                keypoints_image=pose["keypoints_image"],
                confidences=pose["confidences"],
                keypoints_court=keypoints_court,
            )

            self._pose_histories[track_id].add_observation(obs)

    def get_player_pose_history(self, track_id: int) -> Optional[PlayerPoseHistory]:
        """Get pose history for a player."""
        return self._pose_histories.get(track_id)

    def get_all_pose_histories(self) -> Dict[int, PlayerPoseHistory]:
        """Get all pose histories."""
        return self._pose_histories

    def analyze_shot_form(
        self,
        track_id: int,
        start_frame: int,
        end_frame: int,
        fps: float = 30.0,
    ) -> Optional[Dict]:
        """
        Analyze shot form for a specific player during a shot sequence.

        Args:
            track_id: Player track ID
            start_frame: Shot start frame
            end_frame: Shot end frame
            fps: Video frame rate

        Returns:
            Dict with shot form analysis
        """
        history = self._pose_histories.get(track_id)
        if history is None:
            return None

        # Filter observations to shot window
        shot_obs = [
            obs for obs in history.observations
            if start_frame <= obs.frame_idx <= end_frame
        ]

        if len(shot_obs) < 3:
            return None

        # Analyze joint sequences
        analysis = {
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": len(shot_obs),
        }

        # Get release point estimate
        release = history.get_release_point_estimate(hand="right", fps=fps)
        if release:
            analysis["release_point"] = release

        # Compute average joint positions during shot
        avg_positions = {}
        for joint in [JointID.RIGHT_WRIST, JointID.RIGHT_ELBOW, JointID.RIGHT_SHOULDER]:
            frames, positions = history.get_joint_trajectory(joint, as_court=False)
            shot_mask = (frames >= start_frame) & (frames <= end_frame)
            if np.any(shot_mask):
                avg_positions[joint.name] = np.mean(positions[shot_mask], axis=0).tolist()

        analysis["avg_joint_positions"] = avg_positions

        # Elbow angle estimation (simple 2D angle)
        if all(j.name in avg_positions for j in [JointID.RIGHT_WRIST, JointID.RIGHT_ELBOW, JointID.RIGHT_SHOULDER]):
            wrist = np.array(avg_positions["RIGHT_WRIST"])
            elbow = np.array(avg_positions["RIGHT_ELBOW"])
            shoulder = np.array(avg_positions["RIGHT_SHOULDER"])

            # Vectors
            v1 = wrist - elbow
            v2 = shoulder - elbow

            # Angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
            angle_deg = np.degrees(angle_rad)
            analysis["elbow_angle_deg"] = float(angle_deg)

        return analysis

    def reset(self) -> None:
        """Reset all pose histories."""
        self._pose_histories.clear()


def create_pose_pipeline(cfg=None) -> PosePipeline:
    """
    Create PosePipeline from config.

    Args:
        cfg: CVConfig instance

    Returns:
        Configured PosePipeline
    """
    # Default to lightweight model
    model_name = "yolov8n-pose"

    if cfg is not None:
        model_name = getattr(cfg, "pose_model_name", "yolov8n-pose")
        conf = getattr(cfg, "confidence_threshold", 0.3)
        iou = getattr(cfg, "iou_threshold", 0.7)
        return PosePipeline(
            model_name=model_name,
            confidence_threshold=conf,
            iou_threshold=iou,
        )

    return PosePipeline(model_name=model_name)
