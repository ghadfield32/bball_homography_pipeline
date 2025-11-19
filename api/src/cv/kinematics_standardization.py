# api/src/cv/kinematics_standardization.py
"""
Kinematics Standardization Module

Provides a unified coordinate system and skeleton format that aligns with:
- SPL-Open-Data (MLSE's biomechanics datasets)
- mplbasketball visualization tools
- Your CV pipeline (tracking, pose, homography)

Coordinate Systems:
1. Image Space: (u_px, v_px) - pixels, origin top-left
2. Court Space: (x_court_m, y_court_m) - metres, origin court center
3. World Space: (x_world_m, y_world_m, z_world_m) - metres, origin court center, z up

Usage:
    from api.src.cv.kinematics_standardization import (
        JointCoordinate,
        KinematicsStandardizer,
        CANONICAL_JOINTS,
        COCO_TO_CANONICAL,
    )

    standardizer = KinematicsStandardizer(cfg)

    # Convert pose detections to standardized format
    joints = standardizer.standardize_frame(
        frame_idx=0,
        player_id=1,
        keypoints=pose_keypoints,  # COCO format
        homography=H,
    )

    # Export to parquet
    standardizer.export_to_parquet(joints, "output.parquet")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CANONICAL JOINT DEFINITIONS
# =============================================================================

# Canonical joint names aligned with Visual3D / SPL conventions
CANONICAL_JOINTS = [
    "HEAD",
    "NECK",
    "R_SHOULDER",
    "L_SHOULDER",
    "R_ELBOW",
    "L_ELBOW",
    "R_WRIST",
    "L_WRIST",
    "R_HIP",
    "L_HIP",
    "R_KNEE",
    "L_KNEE",
    "R_ANKLE",
    "L_ANKLE",
    # Extended joints (optional)
    "CHEST",
    "PELVIS",
    "R_HAND",
    "L_HAND",
    "BALL",
]

# COCO 17 keypoint indices to canonical joint mapping
COCO_TO_CANONICAL = {
    0: "HEAD",        # nose -> HEAD (approximate)
    5: "L_SHOULDER",
    6: "R_SHOULDER",
    7: "L_ELBOW",
    8: "R_ELBOW",
    9: "L_WRIST",
    10: "R_WRIST",
    11: "L_HIP",
    12: "R_HIP",
    13: "L_KNEE",
    14: "R_KNEE",
    15: "L_ANKLE",
    16: "R_ANKLE",
}

# COCO keypoint indices for computing HEAD from face landmarks
COCO_HEAD_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear

# Visual3D / SPL joint name mapping (common abbreviations)
SPL_TO_CANONICAL = {
    "RSHO": "R_SHOULDER",
    "LSHO": "L_SHOULDER",
    "REL": "R_ELBOW",
    "LEL": "L_ELBOW",
    "RWR": "R_WRIST",
    "LWR": "L_WRIST",
    "RHIP": "R_HIP",
    "LHIP": "L_HIP",
    "RKNE": "R_KNEE",
    "LKNE": "L_KNEE",
    "RANK": "R_ANKLE",
    "LANK": "L_ANKLE",
    "HEAD": "HEAD",
    "NECK": "NECK",
    # Alternative naming
    "R_SHOULDER": "R_SHOULDER",
    "L_SHOULDER": "L_SHOULDER",
}


# =============================================================================
# COURT DIMENSIONS
# =============================================================================

@dataclass
class CourtDimensions:
    """Court dimensions for different leagues."""
    name: str
    length_m: float
    width_m: float
    three_point_radius_m: float
    free_throw_line_m: float  # Distance from basket

    # Common court dimensions
    @classmethod
    def nba(cls) -> "CourtDimensions":
        return cls(
            name="NBA",
            length_m=28.65,  # 94 ft
            width_m=15.24,   # 50 ft
            three_point_radius_m=7.24,  # 23.75 ft
            free_throw_line_m=4.57,  # 15 ft
        )

    @classmethod
    def ncaa(cls) -> "CourtDimensions":
        return cls(
            name="NCAA",
            length_m=28.65,  # 94 ft
            width_m=15.24,   # 50 ft
            three_point_radius_m=6.75,  # 22.15 ft (changed 2019)
            free_throw_line_m=4.57,
        )

    @classmethod
    def fiba(cls) -> "CourtDimensions":
        return cls(
            name="FIBA",
            length_m=28.0,
            width_m=15.0,
            three_point_radius_m=6.75,
            free_throw_line_m=4.60,
        )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class JointCoordinate:
    """
    Single joint observation in standardized format.

    This is the atomic unit of our kinematics data schema.
    All coordinates follow the conventions defined in KINEMATICS_STANDARD.md.
    """
    # Identifiers
    video_id: str
    frame_idx: int
    timestamp_s: float
    player_id: int

    # Player metadata
    team_id: int = -1
    jersey_number: Optional[str] = None

    # Joint identity
    joint: str = ""

    # Image coordinates (pixels, origin top-left)
    u_px: float = np.nan
    v_px: float = np.nan

    # Court coordinates (metres, origin court center)
    x_court_m: float = np.nan
    y_court_m: float = np.nan

    # World coordinates (metres, origin court center, z up)
    x_world_m: float = np.nan
    y_world_m: float = np.nan
    z_world_m: float = np.nan

    # Quality metrics
    joint_confidence: float = 0.0
    homography_quality: float = 0.0

    # Metadata
    homography_segment_id: int = 0
    shot_id: Optional[int] = None
    possession_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return asdict(self)


@dataclass
class StandardizedFrame:
    """All joint observations for a single frame."""
    frame_idx: int
    timestamp_s: float
    joints: List[JointCoordinate] = field(default_factory=list)

    # Frame-level metadata
    homography_valid: bool = False
    num_players: int = 0


# =============================================================================
# HEIGHT ESTIMATION
# =============================================================================

# Approximate joint heights as fraction of player height (standing)
# Based on anthropometric data
JOINT_HEIGHT_RATIOS = {
    "HEAD": 0.93,
    "NECK": 0.87,
    "R_SHOULDER": 0.82,
    "L_SHOULDER": 0.82,
    "R_ELBOW": 0.63,
    "L_ELBOW": 0.63,
    "R_WRIST": 0.47,
    "L_WRIST": 0.47,
    "R_HIP": 0.53,
    "L_HIP": 0.53,
    "R_KNEE": 0.29,
    "L_KNEE": 0.29,
    "R_ANKLE": 0.04,
    "L_ANKLE": 0.04,
    "CHEST": 0.75,
    "PELVIS": 0.53,
}

# Average NBA player height in metres
DEFAULT_PLAYER_HEIGHT_M = 2.01  # ~6'7"


def estimate_joint_height(
    joint: str,
    player_height_m: float = DEFAULT_PLAYER_HEIGHT_M,
) -> float:
    """
    Estimate z-coordinate (height) of a joint based on anthropometric ratios.

    This is a rough approximation for when we don't have depth data.
    Assumes player is standing upright.

    Args:
        joint: Canonical joint name
        player_height_m: Player height in metres

    Returns:
        Estimated height in metres above floor
    """
    ratio = JOINT_HEIGHT_RATIOS.get(joint, 0.5)
    return ratio * player_height_m


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

def image_to_court(
    u_px: float,
    v_px: float,
    H: np.ndarray,
    court_dims: Optional[CourtDimensions] = None,
) -> Tuple[float, float]:
    """
    Transform image coordinates to court coordinates using homography.

    Args:
        u_px: Image x-coordinate (pixels)
        v_px: Image y-coordinate (pixels)
        H: 3x3 homography matrix (image -> court)
        court_dims: Court dimensions (for bounds checking)

    Returns:
        (x_court_m, y_court_m) in metres, origin at court center
    """
    if H is None:
        return np.nan, np.nan

    # Homogeneous coordinates
    pt = np.array([u_px, v_px, 1.0])

    # Apply homography
    pt_court = H @ pt

    # Normalize
    if abs(pt_court[2]) < 1e-8:
        return np.nan, np.nan

    x_court = pt_court[0] / pt_court[2]
    y_court = pt_court[1] / pt_court[2]

    # Bounds check (optional)
    if court_dims is not None:
        half_length = court_dims.length_m / 2
        half_width = court_dims.width_m / 2

        if abs(x_court) > half_length * 1.5 or abs(y_court) > half_width * 1.5:
            # Point is way off court, likely bad homography
            return np.nan, np.nan

    return float(x_court), float(y_court)


def court_to_world(
    x_court_m: float,
    y_court_m: float,
    z_estimate_m: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Convert court 2D coordinates to world 3D coordinates.

    For now, this is trivial (court plane is at z=0).
    The z_estimate allows setting height for non-floor joints.

    Args:
        x_court_m: Court x-coordinate (metres)
        y_court_m: Court y-coordinate (metres)
        z_estimate_m: Height estimate (metres)

    Returns:
        (x_world_m, y_world_m, z_world_m)
    """
    return float(x_court_m), float(y_court_m), float(z_estimate_m)


def world_to_court(
    x_world_m: float,
    y_world_m: float,
    z_world_m: float,
) -> Tuple[float, float]:
    """
    Project world 3D coordinates to court 2D (drop z).

    Args:
        x_world_m, y_world_m, z_world_m: World coordinates

    Returns:
        (x_court_m, y_court_m)
    """
    return float(x_world_m), float(y_world_m)


# =============================================================================
# MAIN STANDARDIZER CLASS
# =============================================================================

@dataclass
class KinematicsStandardizer:
    """
    Main class for converting CV pipeline output to standardized format.

    Handles:
    - Coordinate transformations (image -> court -> world)
    - Joint name mapping (COCO -> canonical)
    - Height estimation
    - Data export to parquet/CSV
    """
    # Court configuration
    court_dims: CourtDimensions = field(default_factory=CourtDimensions.nba)

    # Default player height for z estimation
    default_player_height_m: float = DEFAULT_PLAYER_HEIGHT_M

    # Whether to estimate z from anthropometric ratios
    estimate_z_from_ratios: bool = True

    # Units for output (internal is always metres)
    output_units: str = "metres"  # "metres" or "feet"

    # Conversion factor
    _metres_to_feet: float = 3.28084

    def standardize_keypoints(
        self,
        video_id: str,
        frame_idx: int,
        timestamp_s: float,
        player_id: int,
        keypoints: np.ndarray,
        keypoint_confidences: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        team_id: int = -1,
        jersey_number: Optional[str] = None,
        shot_id: Optional[int] = None,
        homography_segment_id: int = 0,
        homography_quality: float = 0.0,
        player_height_m: Optional[float] = None,
    ) -> List[JointCoordinate]:
        """
        Convert COCO keypoints to standardized JointCoordinate list.

        Args:
            video_id: Video identifier
            frame_idx: Frame number
            timestamp_s: Timestamp in seconds
            player_id: Player/track ID
            keypoints: (17, 2) or (17, 3) array of COCO keypoints
            keypoint_confidences: (17,) array of confidences
            H: Homography matrix (image -> court)
            team_id: Team identifier
            jersey_number: Jersey number string
            shot_id: Shot event ID
            homography_segment_id: Camera segment ID
            homography_quality: Homography RMSE
            player_height_m: Player height for z estimation

        Returns:
            List of JointCoordinate objects
        """
        joints = []
        height = player_height_m or self.default_player_height_m

        # Handle HEAD specially (average of face landmarks)
        head_pts = []
        head_confs = []

        for coco_idx, canonical_name in COCO_TO_CANONICAL.items():
            if coco_idx >= len(keypoints):
                continue

            kp = keypoints[coco_idx]
            u_px, v_px = float(kp[0]), float(kp[1])

            conf = 1.0
            if keypoint_confidences is not None and coco_idx < len(keypoint_confidences):
                conf = float(keypoint_confidences[coco_idx])

            # Collect head landmarks
            if coco_idx in COCO_HEAD_INDICES:
                if conf > 0.1:
                    head_pts.append((u_px, v_px))
                    head_confs.append(conf)

            # Skip individual head landmarks (we'll use averaged HEAD)
            if coco_idx in COCO_HEAD_INDICES and coco_idx != 0:
                continue

            # Transform to court coordinates
            x_court, y_court = image_to_court(u_px, v_px, H, self.court_dims)

            # Estimate z from anthropometric ratios
            if self.estimate_z_from_ratios:
                z_est = estimate_joint_height(canonical_name, height)
            else:
                z_est = 0.0

            # Convert to world coordinates
            x_world, y_world, z_world = court_to_world(x_court, y_court, z_est)

            joint = JointCoordinate(
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                player_id=player_id,
                team_id=team_id,
                jersey_number=jersey_number,
                joint=canonical_name,
                u_px=u_px,
                v_px=v_px,
                x_court_m=x_court,
                y_court_m=y_court,
                x_world_m=x_world,
                y_world_m=y_world,
                z_world_m=z_world,
                joint_confidence=conf,
                homography_quality=homography_quality,
                homography_segment_id=homography_segment_id,
                shot_id=shot_id,
            )

            # Update HEAD with averaged position
            if canonical_name == "HEAD" and head_pts:
                avg_u = np.mean([p[0] for p in head_pts])
                avg_v = np.mean([p[1] for p in head_pts])
                avg_conf = np.mean(head_confs)

                x_court, y_court = image_to_court(avg_u, avg_v, H, self.court_dims)
                z_est = estimate_joint_height("HEAD", height)
                x_world, y_world, z_world = court_to_world(x_court, y_court, z_est)

                joint.u_px = avg_u
                joint.v_px = avg_v
                joint.x_court_m = x_court
                joint.y_court_m = y_court
                joint.x_world_m = x_world
                joint.y_world_m = y_world
                joint.z_world_m = z_world
                joint.joint_confidence = avg_conf

            joints.append(joint)

        return joints

    def to_dataframe(
        self,
        joints: List[JointCoordinate],
        convert_units: bool = True,
    ) -> pd.DataFrame:
        """
        Convert list of JointCoordinates to pandas DataFrame.

        Args:
            joints: List of JointCoordinate objects
            convert_units: Whether to convert to output_units

        Returns:
            DataFrame with standardized columns
        """
        if not joints:
            return pd.DataFrame()

        df = pd.DataFrame([j.to_dict() for j in joints])

        # Convert units if needed
        if convert_units and self.output_units == "feet":
            for col in ["x_court_m", "y_court_m", "x_world_m", "y_world_m", "z_world_m"]:
                if col in df.columns:
                    new_col = col.replace("_m", "_ft")
                    df[new_col] = df[col] * self._metres_to_feet

        return df

    def export_to_parquet(
        self,
        joints: List[JointCoordinate],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export standardized joints to parquet file.

        Args:
            joints: List of JointCoordinate objects
            output_path: Output file path
            metadata: Additional metadata to store

        Returns:
            Path to created file
        """
        df = self.to_dataframe(joints)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Store metadata as parquet metadata
        if metadata:
            df.attrs["metadata"] = json.dumps(metadata)

        df.to_parquet(output_path, index=False)

        return output_path

    def export_to_csv(
        self,
        joints: List[JointCoordinate],
        output_path: Path,
    ) -> Path:
        """Export standardized joints to CSV file."""
        df = self.to_dataframe(joints)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        return output_path


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_kinematics_standardizer(cfg=None) -> KinematicsStandardizer:
    """
    Create KinematicsStandardizer from config.

    Args:
        cfg: CVConfig instance

    Returns:
        Configured KinematicsStandardizer
    """
    if cfg is None:
        return KinematicsStandardizer()

    # Determine court type
    court_type = getattr(cfg, "kinematics_court_type", "NBA")
    if court_type.upper() == "NBA":
        court_dims = CourtDimensions.nba()
    elif court_type.upper() == "NCAA":
        court_dims = CourtDimensions.ncaa()
    elif court_type.upper() == "FIBA":
        court_dims = CourtDimensions.fiba()
    else:
        court_dims = CourtDimensions.nba()

    return KinematicsStandardizer(
        court_dims=court_dims,
        default_player_height_m=getattr(cfg, "default_player_height_m", DEFAULT_PLAYER_HEIGHT_M),
        estimate_z_from_ratios=getattr(cfg, "estimate_z_from_ratios", True),
        output_units=getattr(cfg, "kinematics_output_units", "metres"),
    )
