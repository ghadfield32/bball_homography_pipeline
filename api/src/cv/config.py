"""
Central config for the CV shot pipeline.
- Keeps all IDs, thresholds, and paths in one place
- Never hardcodes API keys; uses env vars (ROBOFLOW_API_KEY or INFERENCE_API_KEY)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional, Tuple
import supervision as sv
from sports import MeasurementUnit
from sports.basketball import CourtConfiguration, League

@dataclass(frozen=True)
class CVConfig:
    # --- Workspace & IO ---
    workspace_dir: Path = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
    data_dir: Path = workspace_dir / "api/src/cv/data"

    video_dir: Path = data_dir / "video"
    output_dir: Path = video_dir / "outputs"

    source_video_path: Path = Path(
        os.getenv(
            "SOURCE_VIDEO_PATH",
            "/workspace/api/src/cv/data/video/boston-celtics-new-york-knicks-game-1/"
            "boston-celtics-new-york-knicks-game-1-q1-03.16-03.11.mp4",
        )
    )

    # Optional image input (single-still pipeline)
    images_dir: Path = data_dir / "images"
    image_output_dir: Path = images_dir / "outputs"

    # --- Models (Roboflow Inference) ---
    player_model_id: str = os.getenv(
        "PLAYER_DETECTION_MODEL_ID", "basketball-player-detection-3-ycjdo/4"
    )
    court_model_id: str = os.getenv(
        "COURT_DETECTION_MODEL_ID", "basketball-court-detection-2/14"
    )

    # --- Thresholds ---
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.70
    keypoint_conf_threshold: float = 0.50
    detection_confidence_court: float = 0.30
    min_keypoints_required: int = 4

    # Homography quality gates
    homography_rmse_court_max: float = float(os.getenv("H_RMSE_COURT_MAX", "1.5"))  # feet
    homography_rmse_image_max: float = float(os.getenv("H_RMSE_IMAGE_MAX", "5.0"))  # pixels
    use_cached_transform_frames: int = int(os.getenv("USE_CACHED_TRANSFORM_FRAMES", "99999"))

    # --- Event logic ---
    reset_time_seconds: float = 1.7
    min_between_starts_seconds: float = 0.5
    cooldown_after_made_seconds: float = 0.5

    # --- Classes ---
    BALL_IN_BASKET_CLASS_ID: int = 1
    JUMP_SHOT_CLASS_ID: int = 5
    LAYUP_DUNK_CLASS_ID: int = 6

    # Optional referee hints (id and/or label match if your model exposes them)
    referee_class_ids: Tuple[int, ...] = tuple(int(x) for x in os.getenv("REFEREE_CLASS_IDS", "").split(",") if x.strip().isdigit())
    referee_labels: Tuple[str, ...] = tuple(x.strip().lower() for x in os.getenv("REFEREE_LABELS", "referee,official").split(",") if x.strip())

    # --- Court drawing ---
    court_scale: int = 20
    court_padding: int = 50
    court_line_thickness: int = 4

    # --- Smoothing ---
    keypoint_smoothing_len: int = 3

    # --- Colors ---
    palette: sv.ColorPalette = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
    ])
    magenta: sv.Color = sv.Color.from_hex("#FF1493")
    cyan: sv.Color = sv.Color.from_hex("#00BFFF")

    # Players (teams) + referee colors
    team_a_color: sv.Color = sv.Color.from_hex("#1F77B4")  # blue
    team_b_color: sv.Color = sv.Color.from_hex("#FF7F0E")  # orange
    referee_color: sv.Color = sv.Color.from_hex("#808080") # grey

    # Shots (made/miss)
    attempt_color: sv.Color = sv.Color.from_hex("#1F77B4")  # legacy fallback
    made_color: sv.Color = sv.Color.from_hex("#007A33")     # green
    miss_color: sv.Color = sv.Color.from_hex("#850101")     # red

    # --- League/Court ---
    court_config: CourtConfiguration = CourtConfiguration(
        league=League.NBA, measurement_unit=MeasurementUnit.FEET
    )

    # --- Smoke test runtime ---
    smoke_max_frames: Optional[int] = 200  # None = full video

    # --- Tracking configuration (tuned for basketball) ---
    enable_tracking: bool = os.getenv("ENABLE_TRACKING", "1") == "1"
    track_activation_threshold: float = float(os.getenv("TRACK_ACTIVATION_THRESHOLD", "0.20"))  # Lower for partial occlusions
    lost_track_buffer: int = int(os.getenv("LOST_TRACK_BUFFER", "60"))  # ~2s for drives/screens
    minimum_matching_threshold: float = float(os.getenv("MINIMUM_MATCHING_THRESHOLD", "0.6"))  # Lower for fast motion
    minimum_consecutive_frames: int = int(os.getenv("MINIMUM_CONSECUTIVE_FRAMES", "2"))  # Reduce flickering

    # --- Segment-level homography ---
    enable_segment_homography: bool = os.getenv("ENABLE_SEGMENT_HOMOGRAPHY", "0") == "1"
    segment_min_frames: int = int(os.getenv("SEGMENT_MIN_FRAMES", "10"))
    segment_change_threshold: float = float(os.getenv("SEGMENT_CHANGE_THRESHOLD", "50.0"))

    # --- Pose estimation ---
    enable_pose_estimation: bool = os.getenv("ENABLE_POSE_ESTIMATION", "0") == "1"
    pose_model_name: str = os.getenv("POSE_MODEL_NAME", "yolov8n-pose")

    # --- Video frame rate ---
    video_fps: int = int(os.getenv("VIDEO_FPS", "30"))

    # --- Debug / Smoke options ---
    start_frame_index: int = int(os.getenv("START_FRAME_INDEX", "65"))
    save_debug_stage_images: bool = os.getenv("SAVE_DEBUG_STAGE_IMAGES", "1") == "1"
    enable_ffmpeg_compression: bool = os.getenv("ENABLE_FFMPEG_COMPRESSION", "0") == "1"

    # --- Event-frame saving ---
    save_event_frames: bool = os.getenv("SAVE_EVENT_FRAMES", "1") == "1"
    event_frame_limit: int = int(os.getenv("EVENT_FRAME_LIMIT", "12"))

    # --- UI / behavior toggles ---
    show_images: bool = os.getenv("SHOW_IMAGES", "0") == "1"

    # Fail fast by default
    strict_fail: bool = os.getenv("STRICT_FAIL", "1") == "1"

    # Write a final single PNG summary (overhead court) per video
    emit_summary_image: bool = os.getenv("EMIT_SUMMARY_IMAGE", "1") == "1"

    preview_images_in_terminal: bool = os.getenv("PREVIEW_IMAGES_IN_TERMINAL", "0") == "1"
    terminal_preview_max_width: int = int(os.getenv("TERMINAL_PREVIEW_MAX_WIDTH", "80"))

    # --- add these fields inside CVConfig dataclass ---
    # Homography robustness / debugging
    enable_robust_homography: bool = os.getenv("ENABLE_ROBUST_HOMO", "1") == "1"
    ransac_reproj_thresh_px: float = float(os.getenv("RANSAC_REPROJ_THRESH_PX", "10.0"))
    min_inlier_ratio: float = float(os.getenv("HOMO_MIN_INLIER_RATIO", "0.5"))  # relative to kept keypoints
    min_spread_px: float = float(os.getenv("HOMO_MIN_SPREAD_PX", "80.0"))       # ensure not degenerate cluster

    # Homography RANSAC thresholds (destination-units explicit)
    ransac_reproj_thresh_court_ft: float = float(os.getenv("RANSAC_REPROJ_THRESH_COURT_FT", "1.0"))
    ransac_reproj_thresh_image_px: float = float(os.getenv("RANSAC_REPROJ_THRESH_IMAGE_PX", "5.0"))

    # --- Homography proof / visualization ---
    homography_proof_enable: bool = os.getenv("HOMOGRAPHY_PROOF_ENABLE", "1") == "1"
    homography_grid_step_ft: float = float(os.getenv("HOMOGRAPHY_GRID_STEP_FT", "5.0"))

# ---- Helpers ----

def ensure_dirs(cfg: CVConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def resolve_api_key() -> str:
    """Return an API key from env, supporting either ROBOFLOW_API_KEY or INFERENCE_API_KEY."""
    api_key = os.getenv("ROBOFLOW_API_KEY") or os.getenv("INFERENCE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set ROBOFLOW_API_KEY or INFERENCE_API_KEY in the environment."
        )
    return api_key


def _set_inference_feature_flags() -> None:
    """
    Disable optional models we do not use so 'inference' doesn't warn about missing extras.
    This does NOT affect the player/court models.
    """
    os.environ.setdefault("QWEN_2_5_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_SAM2_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_CLIP_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
    os.environ.setdefault("SMOLVLM2_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_GROUNDINGDINO_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_PE_ENABLED", "False")


def load_models(cfg: CVConfig):
    """Lazy import to avoid heavy import at module import time."""
    _set_inference_feature_flags()
    from inference import get_model  # type: ignore

    api_key = resolve_api_key()
    player_model = get_model(model_id=cfg.player_model_id, api_key=api_key)
    court_model = get_model(model_id=cfg.court_model_id, api_key=api_key)
    return player_model, court_model


def court_base_image(cfg: CVConfig):
    from sports.basketball import draw_court

    base = draw_court(
        config=cfg.court_config,
        scale=cfg.court_scale,
        padding=cfg.court_padding,
        line_thickness=cfg.court_line_thickness,
    )
    return base
