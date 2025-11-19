# api/src/biomech/spl_adapter.py
"""
SPL-Open-Data Adapter

Converts MLSE SPL-Open-Data biomechanics data to the standardized format
used by our CV pipeline, enabling unified analysis and visualization.

SPL-Open-Data: https://github.com/mlsedigital/SPL-Open-Data
- Basketball free-throw dataset (125 trials, 1 participant)
- Motion capture data from Visual3D pipeline

This adapter:
1. Loads SPL data (C3D/CSV exports)
2. Maps SPL joint names to canonical names
3. Transforms SPL lab coordinates to court-centric world coordinates
4. Exports in our standardized schema

Usage:
    from api.src.biomech.spl_adapter import SPLAdapter

    adapter = SPLAdapter()
    adapter.set_transform_from_landmarks(spl_ankle_pos, court_ankle_pos)

    joints = adapter.load_trial("path/to/trial.csv", trial_id=1)
    df = adapter.to_dataframe(joints)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from api.src.cv.kinematics_standardization import (
    JointCoordinate,
    SPL_TO_CANONICAL,
    CANONICAL_JOINTS,
    CourtDimensions,
)


# =============================================================================
# SPL COORDINATE TRANSFORMATION
# =============================================================================

@dataclass
class SPLTransform:
    """
    Rigid transformation from SPL lab coordinates to world coordinates.

    Solves: X_world = R @ X_spl + t

    The SPL lab likely uses a different coordinate frame than our court-centric
    world frame. This class handles the rotation and translation.
    """
    # 3x3 rotation matrix
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    # 3x1 translation vector
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Scale factor (in case units differ)
    scale: float = 1.0  # mm to metres = 0.001

    def transform(self, X_spl: np.ndarray) -> np.ndarray:
        """
        Transform SPL coordinates to world coordinates.

        Args:
            X_spl: (3,) or (N, 3) array of SPL coordinates

        Returns:
            World coordinates in same shape
        """
        X_spl = np.atleast_2d(X_spl) * self.scale
        X_world = (self.R @ X_spl.T).T + self.t
        return X_world.squeeze()

    def inverse_transform(self, X_world: np.ndarray) -> np.ndarray:
        """
        Transform world coordinates back to SPL coordinates.

        Args:
            X_world: (3,) or (N, 3) array of world coordinates

        Returns:
            SPL coordinates
        """
        X_world = np.atleast_2d(X_world)
        X_spl = (self.R.T @ (X_world - self.t).T).T / self.scale
        return X_spl.squeeze()

    @classmethod
    def from_landmarks(
        cls,
        spl_points: np.ndarray,
        world_points: np.ndarray,
    ) -> "SPLTransform":
        """
        Compute transform from corresponding landmarks (Kabsch algorithm).

        Args:
            spl_points: (N, 3) array of SPL coordinates
            world_points: (N, 3) array of corresponding world coordinates

        Returns:
            SPLTransform with fitted R, t
        """
        spl_points = np.atleast_2d(spl_points)
        world_points = np.atleast_2d(world_points)

        # Compute centroids
        spl_centroid = np.mean(spl_points, axis=0)
        world_centroid = np.mean(world_points, axis=0)

        # Center the points
        spl_centered = spl_points - spl_centroid
        world_centered = world_points - world_centroid

        # Compute covariance matrix
        H = spl_centered.T @ world_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = world_centroid - R @ spl_centroid

        return cls(R=R, t=t, scale=1.0)


# =============================================================================
# DEFAULT SPL TO WORLD TRANSFORM
# =============================================================================

def get_default_spl_transform() -> SPLTransform:
    """
    Get default transform for SPL free-throw data.

    This is a placeholder - you should compute the actual transform
    by matching landmarks (ankle positions, hoop location, etc.)
    once you have the SPL data locally.

    Assumptions for default:
    - SPL uses mm, we use metres (scale = 0.001)
    - SPL X = forward (towards basket) = our X_world
    - SPL Y = left = our -Y_world
    - SPL Z = up = our Z_world
    - Origin offset: SPL origin is at free-throw line
    """
    # Rotation: flip Y axis
    R = np.array([
        [1,  0, 0],
        [0, -1, 0],
        [0,  0, 1],
    ], dtype=float)

    # Translation: move origin from free-throw line to court center
    # Free-throw line is ~9.75m from court center (half-length - 4.57m from basket)
    t = np.array([-9.75, 0, 0])

    return SPLTransform(R=R, t=t, scale=0.001)


# =============================================================================
# SPL DATA LOADER
# =============================================================================

@dataclass
class SPLAdapter:
    """
    Adapter for loading and converting SPL-Open-Data to standardized format.

    Handles:
    - Loading CSV/C3D exports from Visual3D
    - Joint name mapping
    - Coordinate transformation
    - Export to standardized schema
    """
    # Coordinate transform
    transform: SPLTransform = field(default_factory=get_default_spl_transform)

    # Court dimensions
    court_dims: CourtDimensions = field(default_factory=CourtDimensions.nba)

    # Frame rate (typical for mocap)
    fps: float = 120.0

    # Dataset metadata
    dataset_id: str = "SPL_FREETHROW"
    participant_id: int = 1

    def set_transform_from_landmarks(
        self,
        spl_landmarks: np.ndarray,
        world_landmarks: np.ndarray,
    ) -> None:
        """
        Set coordinate transform from corresponding landmarks.

        Args:
            spl_landmarks: (N, 3) SPL coordinates of known points
            world_landmarks: (N, 3) world coordinates of same points
        """
        self.transform = SPLTransform.from_landmarks(spl_landmarks, world_landmarks)

    def load_trial_csv(
        self,
        csv_path: Path,
        trial_id: int = 0,
    ) -> List[JointCoordinate]:
        """
        Load a single trial from CSV export.

        Expected CSV format (typical Visual3D export):
        - Columns: Frame, Time, RSHO_X, RSHO_Y, RSHO_Z, LSHO_X, ...
        - One row per frame

        Args:
            csv_path: Path to CSV file
            trial_id: Trial identifier

        Returns:
            List of JointCoordinate objects
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        joints = []

        # Find all joint columns
        joint_columns = {}
        for col in df.columns:
            # Look for patterns like "RSHO_X", "RSHO_Y", "RSHO_Z"
            for spl_name in SPL_TO_CANONICAL.keys():
                if col.startswith(spl_name + "_") or col.startswith(spl_name.upper() + "_"):
                    axis = col[-1].upper()  # X, Y, or Z
                    if spl_name not in joint_columns:
                        joint_columns[spl_name] = {}
                    joint_columns[spl_name][axis] = col

        # Process each frame
        for idx, row in df.iterrows():
            frame_idx = int(row.get("Frame", idx))
            timestamp_s = row.get("Time", frame_idx / self.fps)

            # Process each joint
            for spl_name, cols in joint_columns.items():
                if "X" not in cols or "Y" not in cols or "Z" not in cols:
                    continue

                canonical_name = SPL_TO_CANONICAL.get(spl_name, spl_name)
                if canonical_name not in CANONICAL_JOINTS:
                    continue

                # Get SPL coordinates
                x_spl = row[cols["X"]]
                y_spl = row[cols["Y"]]
                z_spl = row[cols["Z"]]

                if pd.isna(x_spl) or pd.isna(y_spl) or pd.isna(z_spl):
                    continue

                # Transform to world coordinates
                spl_pos = np.array([x_spl, y_spl, z_spl])
                world_pos = self.transform.transform(spl_pos)

                x_world, y_world, z_world = world_pos

                # Court coordinates (drop z)
                x_court, y_court = x_world, y_world

                joint = JointCoordinate(
                    video_id=f"{self.dataset_id}_trial{trial_id:03d}",
                    frame_idx=frame_idx,
                    timestamp_s=float(timestamp_s),
                    player_id=self.participant_id,
                    team_id=0,
                    jersey_number=None,
                    joint=canonical_name,
                    u_px=np.nan,  # No image coordinates for mocap
                    v_px=np.nan,
                    x_court_m=float(x_court),
                    y_court_m=float(y_court),
                    x_world_m=float(x_world),
                    y_world_m=float(y_world),
                    z_world_m=float(z_world),
                    joint_confidence=1.0,  # Mocap is high confidence
                    homography_quality=1.0,
                    shot_id=trial_id,
                )

                joints.append(joint)

        return joints

    def load_multiple_trials(
        self,
        data_dir: Path,
        pattern: str = "*.csv",
    ) -> List[JointCoordinate]:
        """
        Load all trials from a directory.

        Args:
            data_dir: Directory containing trial CSV files
            pattern: Glob pattern for CSV files

        Returns:
            List of JointCoordinate objects from all trials
        """
        data_dir = Path(data_dir)
        all_joints = []

        for i, csv_path in enumerate(sorted(data_dir.glob(pattern))):
            trial_joints = self.load_trial_csv(csv_path, trial_id=i)
            all_joints.extend(trial_joints)

        return all_joints

    def to_dataframe(self, joints: List[JointCoordinate]) -> pd.DataFrame:
        """
        Convert joint list to DataFrame.

        Args:
            joints: List of JointCoordinate objects

        Returns:
            DataFrame with standardized columns
        """
        if not joints:
            return pd.DataFrame()

        return pd.DataFrame([j.to_dict() for j in joints])

    def export_to_parquet(
        self,
        joints: List[JointCoordinate],
        output_path: Path,
    ) -> Path:
        """
        Export joints to parquet file.

        Args:
            joints: List of JointCoordinate objects
            output_path: Output file path

        Returns:
            Path to created file
        """
        df = self.to_dataframe(joints)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False)

        return output_path

    def get_trial_summary(self, joints: List[JointCoordinate]) -> Dict[str, Any]:
        """
        Get summary statistics for loaded trials.

        Args:
            joints: List of JointCoordinate objects

        Returns:
            Dictionary with summary statistics
        """
        if not joints:
            return {}

        df = self.to_dataframe(joints)

        return {
            "num_frames": df["frame_idx"].nunique(),
            "num_joints": df["joint"].nunique(),
            "num_trials": df["shot_id"].nunique(),
            "joints": sorted(df["joint"].unique().tolist()),
            "duration_s": df["timestamp_s"].max() - df["timestamp_s"].min(),
            "z_range_m": (df["z_world_m"].min(), df["z_world_m"].max()),
        }


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_spl_transform(
    adapter: SPLAdapter,
    spl_points: np.ndarray,
    expected_world: np.ndarray,
    tolerance_m: float = 0.01,
) -> bool:
    """
    Validate that the SPL transform correctly maps test points.

    Args:
        adapter: SPLAdapter with transform set
        spl_points: (N, 3) test SPL coordinates
        expected_world: (N, 3) expected world coordinates
        tolerance_m: Maximum allowed error in metres

    Returns:
        True if all points are within tolerance
    """
    transformed = np.array([adapter.transform.transform(p) for p in spl_points])
    errors = np.linalg.norm(transformed - expected_world, axis=1)

    return np.all(errors < tolerance_m)


def compute_alignment_error(
    adapter: SPLAdapter,
    spl_points: np.ndarray,
    expected_world: np.ndarray,
) -> Dict[str, float]:
    """
    Compute alignment error statistics.

    Args:
        adapter: SPLAdapter with transform set
        spl_points: (N, 3) test SPL coordinates
        expected_world: (N, 3) expected world coordinates

    Returns:
        Dictionary with error statistics
    """
    transformed = np.array([adapter.transform.transform(p) for p in spl_points])
    errors = np.linalg.norm(transformed - expected_world, axis=1)

    return {
        "mean_error_m": float(np.mean(errors)),
        "max_error_m": float(np.max(errors)),
        "std_error_m": float(np.std(errors)),
        "num_points": len(errors),
    }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_spl_adapter(cfg=None) -> SPLAdapter:
    """
    Create SPLAdapter from config.

    Args:
        cfg: CVConfig instance

    Returns:
        Configured SPLAdapter
    """
    if cfg is None:
        return SPLAdapter()

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

    return SPLAdapter(
        court_dims=court_dims,
        fps=getattr(cfg, "spl_fps", 120.0),
        dataset_id=getattr(cfg, "spl_dataset_id", "SPL_FREETHROW"),
    )
