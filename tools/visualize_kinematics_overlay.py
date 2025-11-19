#!/usr/bin/env python3
"""
Kinematics Visualization Overlay Tool

Overlays joint positions from kinematics parquet onto video frames
for visual sanity checking of the pose → kinematics pipeline.

Usage:
    python -m tools.visualize_kinematics_overlay \\
        --video path/to/video.mp4 \\
        --kinematics path/to/video_kinematics.parquet \\
        --output path/to/output_frames/ \\
        --every-n-frames 10

This tool:
- Loads the exported kinematics parquet
- For sampled frames, draws joints on the video frame
- Verifies joint names are canonical (R_WRIST, L_KNEE, etc.)
- Optionally prints court coordinates for inspection
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


# Joint colors for visualization
JOINT_COLORS = {
    "HEAD": (255, 255, 0),       # Cyan
    "L_SHOULDER": (0, 255, 0),   # Green
    "R_SHOULDER": (0, 255, 0),
    "L_ELBOW": (255, 165, 0),    # Orange
    "R_ELBOW": (255, 165, 0),
    "L_WRIST": (0, 0, 255),      # Red
    "R_WRIST": (0, 0, 255),
    "L_HIP": (255, 0, 255),      # Magenta
    "R_HIP": (255, 0, 255),
    "L_KNEE": (255, 255, 255),   # White
    "R_KNEE": (255, 255, 255),
    "L_ANKLE": (128, 128, 128),  # Gray
    "R_ANKLE": (128, 128, 128),
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    ("L_ANKLE", "L_KNEE"),
    ("L_KNEE", "L_HIP"),
    ("R_ANKLE", "R_KNEE"),
    ("R_KNEE", "R_HIP"),
    ("L_HIP", "R_HIP"),
    ("L_SHOULDER", "L_HIP"),
    ("R_SHOULDER", "R_HIP"),
    ("L_SHOULDER", "R_SHOULDER"),
    ("L_SHOULDER", "L_ELBOW"),
    ("L_ELBOW", "L_WRIST"),
    ("R_SHOULDER", "R_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("L_SHOULDER", "HEAD"),
    ("R_SHOULDER", "HEAD"),
]


def validate_kinematics_schema(df: pd.DataFrame) -> None:
    """
    Validate that kinematics DataFrame has expected columns and joint names.

    Raises:
        ValueError: If schema is invalid
    """
    required_cols = {"frame_idx", "player_id", "joint", "u_px", "v_px"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check joint names are canonical
    expected_joints = {
        "HEAD", "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW",
        "L_WRIST", "R_WRIST", "L_HIP", "R_HIP", "L_KNEE", "R_KNEE",
        "L_ANKLE", "R_ANKLE"
    }
    actual_joints = set(df["joint"].unique())

    unexpected = actual_joints - expected_joints
    if unexpected:
        print(f"[WARN] Unexpected joint names found: {unexpected}")

    missing_joints = expected_joints - actual_joints
    if missing_joints:
        print(f"[INFO] Missing joint names (may be normal): {missing_joints}")


def get_joints_for_frame(
    df: pd.DataFrame,
    frame_idx: int,
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Get all joint positions for a specific frame.

    Returns:
        {player_id: {joint_name: (u_px, v_px)}}
    """
    frame_data = df[df["frame_idx"] == frame_idx]

    result: Dict[int, Dict[str, Tuple[float, float]]] = {}

    for _, row in frame_data.iterrows():
        player_id = int(row["player_id"])
        joint = row["joint"]
        u_px = row["u_px"]
        v_px = row["v_px"]

        if player_id not in result:
            result[player_id] = {}

        if not np.isnan(u_px) and not np.isnan(v_px):
            result[player_id][joint] = (float(u_px), float(v_px))

    return result


def draw_skeleton_on_frame(
    frame: np.ndarray,
    joints: Dict[str, Tuple[float, float]],
    player_id: int,
    draw_connections: bool = True,
) -> np.ndarray:
    """
    Draw a player's skeleton on a frame.

    Args:
        frame: BGR image
        joints: {joint_name: (u_px, v_px)}
        player_id: Player ID for labeling
        draw_connections: Whether to draw skeleton lines

    Returns:
        Annotated frame
    """
    # Draw connections first (so joints draw on top)
    if draw_connections:
        for j1, j2 in SKELETON_CONNECTIONS:
            if j1 in joints and j2 in joints:
                pt1 = (int(joints[j1][0]), int(joints[j1][1]))
                pt2 = (int(joints[j2][0]), int(joints[j2][1]))
                cv2.line(frame, pt1, pt2, (100, 100, 100), 1)

    # Draw joints
    for joint_name, (u, v) in joints.items():
        pt = (int(u), int(v))
        color = JOINT_COLORS.get(joint_name, (255, 255, 255))
        cv2.circle(frame, pt, 4, color, -1)
        cv2.circle(frame, pt, 4, (0, 0, 0), 1)  # Black outline

    # Label player ID near head or first joint
    if "HEAD" in joints:
        label_pt = joints["HEAD"]
    elif joints:
        label_pt = list(joints.values())[0]
    else:
        return frame

    label_pos = (int(label_pt[0]) - 10, int(label_pt[1]) - 15)
    cv2.putText(
        frame, f"P{player_id}", label_pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )

    return frame


def print_frame_stats(
    df: pd.DataFrame,
    frame_idx: int,
    verbose: bool = True,
) -> None:
    """Print statistics for a specific frame."""
    frame_data = df[df["frame_idx"] == frame_idx]

    n_players = frame_data["player_id"].nunique()
    n_joints = len(frame_data)

    print(f"\nFrame {frame_idx}:")
    print(f"  Players: {n_players}")
    print(f"  Joint records: {n_joints}")

    if verbose and "x_court_m" in frame_data.columns:
        # Print sample court coordinates
        sample = frame_data[frame_data["joint"] == "R_WRIST"].head(3)
        if len(sample) > 0:
            print("  Sample R_WRIST court coords:")
            for _, row in sample.iterrows():
                print(f"    P{row['player_id']}: ({row['x_court_m']:.2f}, {row['y_court_m']:.2f})")


def visualize_kinematics(
    video_path: Path,
    kinematics_path: Path,
    output_dir: Path,
    every_n_frames: int = 10,
    max_frames: int = 50,
    draw_connections: bool = True,
    verbose: bool = True,
) -> List[Path]:
    """
    Main visualization function.

    Args:
        video_path: Path to input video
        kinematics_path: Path to kinematics parquet
        output_dir: Directory to save annotated frames
        every_n_frames: Sample every N frames
        max_frames: Maximum number of frames to process
        draw_connections: Whether to draw skeleton lines
        verbose: Print detailed stats

    Returns:
        List of saved frame paths
    """
    # Load kinematics
    print(f"Loading kinematics from {kinematics_path}")
    df = pd.read_parquet(kinematics_path)

    # Validate schema
    validate_kinematics_schema(df)

    print(f"\nKinematics summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Unique frames: {df['frame_idx'].nunique()}")
    print(f"  Unique players: {df['player_id'].nunique()}")
    print(f"  Joint names: {sorted(df['joint'].unique())}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {total_frames} frames at {fps} fps")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    processed = 0
    frame_idx = 0

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            # Get joints for this frame
            joints_by_player = get_joints_for_frame(df, frame_idx)

            if joints_by_player:
                # Draw each player's skeleton
                for player_id, joints in joints_by_player.items():
                    frame = draw_skeleton_on_frame(
                        frame, joints, player_id, draw_connections
                    )

                # Add frame info overlay
                cv2.putText(
                    frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Players: {len(joints_by_player)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

                # Save frame
                out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved_paths.append(out_path)

                if verbose:
                    print_frame_stats(df, frame_idx)

                processed += 1

        frame_idx += 1

    cap.release()

    print(f"\nSaved {len(saved_paths)} annotated frames to {output_dir}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Overlay kinematics joints on video frames"
    )
    parser.add_argument(
        "--video", type=Path, required=True,
        help="Path to input video"
    )
    parser.add_argument(
        "--kinematics", type=Path, required=True,
        help="Path to kinematics parquet file"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("kinematics_debug"),
        help="Output directory for annotated frames"
    )
    parser.add_argument(
        "--every-n-frames", type=int, default=10,
        help="Sample every N frames"
    )
    parser.add_argument(
        "--max-frames", type=int, default=50,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--no-connections", action="store_true",
        help="Don't draw skeleton connections"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    visualize_kinematics(
        video_path=args.video,
        kinematics_path=args.kinematics,
        output_dir=args.output,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        draw_connections=not args.no_connections,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
