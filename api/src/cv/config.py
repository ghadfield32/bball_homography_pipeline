# api/src/cv/config.py
"""
Central config for the CV shot pipeline.

Organized by feature area for easy navigation:
0. Profile & Mode (automation)
1. Workspace & IO
2. Detection Models (Roboflow)
3. Detection Thresholds
4. Tracking (ByteTrack + SAM2)
5. Homography & ViewTransformer
6. Jersey OCR & Player Identity
7. Pose Estimation & Biomechanics
8. Shot Arc Analysis
9. Visual Re-ID (SigLIP)
10. API & WebSocket Streaming
11. Visualization & Debug
12. Shot Event Logic
13. Kinematics Standardization

PROFILES:
    Set CV_PROFILE env var to automatically configure multiple features:
    - "fast_debug": Minimal features for quick iteration
    - "tracking_only": Tracking + homography for tactics analysis
    - "full_biomech": All features for biomechanics research
    - "live_stream": Real-time streaming with basic tracking

USAGE:
    from api.src.cv.config import CVConfig, load_cv_config

    cfg = load_cv_config()  # Loads from env, applies profile, validates
    print(describe_cv_config(cfg))

Environment variables override defaults for all configurable parameters.
Never hardcodes API keys; uses ROBOFLOW_API_KEY or INFERENCE_API_KEY.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import supervision as sv
from sports import MeasurementUnit
from sports.basketball import CourtConfiguration, League


# =============================================================================
# PROFILE PRESETS
# =============================================================================

PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast_debug": {
        # Minimal features for quick iteration and testing
        "enable_tracking": True,
        "enable_segment_homography": True,
        "enable_pose_estimation": False,
        "enable_shot_arc_analysis": False,
        "enable_siglip_reid": False,
        "enable_jersey_ocr": False,
        "enable_websocket_streaming": False,
        "enable_kinematics_export": False,
    },
    "tracking_only": {
        # Tracking + homography for tactics/spacing analysis
        "enable_tracking": True,
        "enable_segment_homography": True,
        "enable_jersey_ocr": True,
        "enable_pose_estimation": False,
        "enable_shot_arc_analysis": False,
        "enable_siglip_reid": False,
        "enable_kinematics_export": False,
    },
    "full_biomech": {
        # All features for biomechanics/shot analysis research
        "enable_tracking": True,
        "enable_segment_homography": True,
        "enable_pose_estimation": True,
        "enable_shot_arc_analysis": True,
        "enable_jersey_ocr": True,
        "enable_siglip_reid": True,
        "enable_kinematics_export": True,
        "enable_shot_form_analysis": True,
    },
    "live_stream": {
        # Real-time streaming with basic tracking
        "enable_tracking": True,
        "enable_segment_homography": True,
        "enable_websocket_streaming": True,
        "enable_pose_estimation": False,
        "enable_shot_arc_analysis": False,
        "enable_siglip_reid": False,
        "enable_jersey_ocr": False,
        "streaming_frame_skip": 2,
    },
    "cv_debug_kinematics": {
        # Debug profile for kinematics validation
        # Use with short clips to verify pose → kinematics pipeline
        "enable_tracking": True,
        "enable_segment_homography": True,
        "enable_pose_estimation": True,
        "enable_kinematics_export": True,
        "enable_shot_arc_analysis": False,
        "enable_siglip_reid": False,
        "enable_jersey_ocr": False,
        "enable_websocket_streaming": False,
    },
}


@dataclass(frozen=True)
class CVConfig:
    # ==========================================================================
    # 0. PROFILE & MODE
    # ==========================================================================
    cv_profile: str = os.getenv("CV_PROFILE", "fast_debug")
    """
    High-level pipeline profile. Options: fast_debug, tracking_only, full_biomech, live_stream
    Profiles automatically set multiple feature flags for common use cases.
    Individual env vars still override profile settings.
    """

    # ==========================================================================
    # 1. WORKSPACE & IO
    # ==========================================================================
    workspace_dir: Path = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
    data_dir: Path = workspace_dir / "api/src/cv/data"
    video_dir: Path = data_dir / "video"
    output_dir: Path = video_dir / "outputs"
    images_dir: Path = data_dir / "images"
    image_output_dir: Path = images_dir / "outputs"

    source_video_path: Path = Path(
        os.getenv(
            "SOURCE_VIDEO_PATH",
            "/workspace/api/src/cv/data/video/boston-celtics-new-york-knicks-game-1/"
            "boston-celtics-new-york-knicks-game-1-q1-03.16-03.11.mp4",
        )
    )

    # Video settings
    video_fps: int = int(os.getenv("VIDEO_FPS", "30"))

    # ==========================================================================
    # 2. DETECTION MODELS (Roboflow Inference)
    # ==========================================================================
    # Core detection models
    player_model_id: str = os.getenv(
        "PLAYER_DETECTION_MODEL_ID", "basketball-player-detection-3-ycjdo/4"
    )
    court_model_id: str = os.getenv(
        "COURT_DETECTION_MODEL_ID", "basketball-court-detection-2/14"
    )
    ball_model_id: str = os.getenv(
        "BALL_MODEL_ID", "basketball-detection/1"
    )

    # Keypoint models (for ViewTransformer-style homography)
    court_keypoint_model_id: str = os.getenv(
        "COURT_KEYPOINT_MODEL_ID", "basketball-court-keypoints/1"
    )
    rim_detection_model_id: str = os.getenv(
        "RIM_DETECTION_MODEL_ID", "basketball-rim-detection/1"
    )

    # Jersey number model (for OCR alternative)
    jersey_number_model_id: str = os.getenv(
        "JERSEY_NUMBER_MODEL_ID", "basketball-jersey-numbers/1"
    )

    # ==========================================================================
    # 3. DETECTION THRESHOLDS
    # ==========================================================================
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.70"))
    keypoint_conf_threshold: float = float(os.getenv("KEYPOINT_CONF_THRESHOLD", "0.50"))
    detection_confidence_court: float = float(os.getenv("COURT_CONFIDENCE", "0.30"))
    ball_confidence_threshold: float = float(os.getenv("BALL_CONFIDENCE", "0.30"))
    min_keypoints_required: int = int(os.getenv("MIN_KEYPOINTS_REQUIRED", "4"))

    # Class IDs for shot detection
    BALL_IN_BASKET_CLASS_ID: int = 1
    JUMP_SHOT_CLASS_ID: int = 5
    LAYUP_DUNK_CLASS_ID: int = 6

    # Referee hints
    referee_class_ids: Tuple[int, ...] = tuple(
        int(x) for x in os.getenv("REFEREE_CLASS_IDS", "").split(",") if x.strip().isdigit()
    )
    referee_labels: Tuple[str, ...] = tuple(
        x.strip().lower() for x in os.getenv("REFEREE_LABELS", "referee,official").split(",") if x.strip()
    )

    # ==========================================================================
    # 4. TRACKING (ByteTrack + SAM2)
    # ==========================================================================
    enable_tracking: bool = os.getenv("ENABLE_TRACKING", "1") == "1"

    # ByteTrack parameters (tuned for basketball)
    track_activation_threshold: float = float(os.getenv("TRACK_ACTIVATION_THRESHOLD", "0.20"))
    lost_track_buffer: int = int(os.getenv("LOST_TRACK_BUFFER", "60"))  # ~2s at 30fps
    minimum_matching_threshold: float = float(os.getenv("MINIMUM_MATCHING_THRESHOLD", "0.6"))
    minimum_consecutive_frames: int = int(os.getenv("MINIMUM_CONSECUTIVE_FRAMES", "2"))

    # SAM2 segmentation tracking (optional, for pixel-accurate masks)
    enable_sam2_tracking: bool = os.getenv("ENABLE_SAM2_TRACKING", "0") == "1"
    sam2_model_size: str = os.getenv("SAM2_MODEL_SIZE", "large")  # tiny, small, base, large
    sam2_points_per_side: int = int(os.getenv("SAM2_POINTS_PER_SIDE", "32"))

    # ==========================================================================
    # 5. HOMOGRAPHY & VIEWTRANSFORMER
    # ==========================================================================
    # Segment-level homography (recommended for stable transforms)
    enable_segment_homography: bool = os.getenv("ENABLE_SEGMENT_HOMOGRAPHY", "0") == "1"
    segment_min_frames: int = int(os.getenv("SEGMENT_MIN_FRAMES", "10"))
    segment_change_threshold: float = float(os.getenv("SEGMENT_CHANGE_THRESHOLD", "50.0"))

    # Robust homography (RANSAC)
    enable_robust_homography: bool = os.getenv("ENABLE_ROBUST_HOMO", "1") == "1"
    ransac_reproj_thresh_px: float = float(os.getenv("RANSAC_REPROJ_THRESH_PX", "10.0"))
    ransac_reproj_thresh_court_ft: float = float(os.getenv("RANSAC_REPROJ_THRESH_COURT_FT", "1.0"))
    ransac_reproj_thresh_image_px: float = float(os.getenv("RANSAC_REPROJ_THRESH_IMAGE_PX", "5.0"))
    min_inlier_ratio: float = float(os.getenv("HOMO_MIN_INLIER_RATIO", "0.5"))
    min_spread_px: float = float(os.getenv("HOMO_MIN_SPREAD_PX", "80.0"))

    # Quality gates
    homography_rmse_court_max: float = float(os.getenv("H_RMSE_COURT_MAX", "1.5"))  # feet
    homography_rmse_image_max: float = float(os.getenv("H_RMSE_IMAGE_MAX", "5.0"))  # pixels
    use_cached_transform_frames: int = int(os.getenv("USE_CACHED_TRANSFORM_FRAMES", "99999"))

    # Semantic constraints (line collinearity, arc radius validation)
    enable_semantic_constraints: bool = os.getenv("ENABLE_SEMANTIC_CONSTRAINTS", "1") == "1"
    line_collinearity_threshold_ft: float = float(os.getenv("LINE_COLLINEARITY_THRESH", "0.5"))
    arc_radius_threshold_ft: float = float(os.getenv("ARC_RADIUS_THRESH", "1.0"))
    three_point_radius_ft: float = 23.75  # NBA standard

    # Visualization
    homography_proof_enable: bool = os.getenv("HOMOGRAPHY_PROOF_ENABLE", "1") == "1"
    homography_grid_step_ft: float = float(os.getenv("HOMOGRAPHY_GRID_STEP_FT", "5.0"))

    # ==========================================================================
    # 6. JERSEY OCR & PLAYER IDENTITY
    # ==========================================================================
    enable_jersey_ocr: bool = os.getenv("ENABLE_JERSEY_OCR", "0") == "1"

    # OCR backend: "easyocr", "paddleocr", "smolvlm2", "yolo_classifier", "none"
    jersey_ocr_type: str = os.getenv("JERSEY_OCR_TYPE", "easyocr")
    jersey_ocr_confidence: float = float(os.getenv("JERSEY_OCR_CONFIDENCE", "0.5"))

    # Jersey region extraction (relative to player bbox)
    jersey_number_region_top: float = float(os.getenv("JERSEY_REGION_TOP", "0.1"))
    jersey_number_region_bottom: float = float(os.getenv("JERSEY_REGION_BOTTOM", "0.5"))
    jersey_number_region_left: float = float(os.getenv("JERSEY_REGION_LEFT", "0.2"))
    jersey_number_region_right: float = float(os.getenv("JERSEY_REGION_RIGHT", "0.8"))

    # SmolVLM2 OCR (VLM-based, more accurate but slower)
    smolvlm2_model_name: str = os.getenv("SMOLVLM2_MODEL", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Majority voting for stable assignments
    jersey_vote_window: int = int(os.getenv("JERSEY_VOTE_WINDOW", "10"))

    # ==========================================================================
    # 7. POSE ESTIMATION & BIOMECHANICS
    # ==========================================================================
    enable_pose_estimation: bool = os.getenv("ENABLE_POSE_ESTIMATION", "0") == "1"

    # YOLO-pose model
    pose_model_name: str = os.getenv("POSE_MODEL_NAME", "yolov8n-pose")  # yolov8n/s/m-pose
    pose_confidence_threshold: float = float(os.getenv("POSE_CONFIDENCE", "0.3"))

    # Biomechanics analysis
    enable_shot_form_analysis: bool = os.getenv("ENABLE_SHOT_FORM", "0") == "1"
    release_point_min_frames: int = int(os.getenv("RELEASE_MIN_FRAMES", "5"))

    # Zone detection (for 3-second violations, lane occupancy)
    enable_zone_detection: bool = os.getenv("ENABLE_ZONE_DETECTION", "0") == "1"
    zone_dwell_threshold_frames: int = int(os.getenv("ZONE_DWELL_THRESH", "90"))  # 3s at 30fps

    # ==========================================================================
    # 8. SHOT ARC ANALYSIS
    # ==========================================================================
    enable_shot_arc_analysis: bool = os.getenv("ENABLE_SHOT_ARC_ANALYSIS", "0") == "1"

    # Trajectory tracking
    arc_min_trajectory_points: int = int(os.getenv("ARC_MIN_POINTS", "5"))
    arc_velocity_window_frames: int = int(os.getenv("ARC_VELOCITY_WINDOW", "3"))
    arc_angle_smoothing: int = int(os.getenv("ARC_ANGLE_SMOOTHING", "3"))

    # Physical constants
    rim_height_ft: float = 10.0
    ball_diameter_ft: float = 0.78  # ~9.4 inches

    # ==========================================================================
    # 9. VISUAL RE-ID (SigLIP Embeddings)
    # ==========================================================================
    enable_siglip_reid: bool = os.getenv("ENABLE_SIGLIP_REID", "0") == "1"

    # SigLIP model
    siglip_model_name: str = os.getenv("SIGLIP_MODEL_NAME", "google/siglip-base-patch16-224")
    siglip_embedding_dim: int = int(os.getenv("SIGLIP_EMBEDDING_DIM", "768"))

    # Matching thresholds
    siglip_similarity_threshold: float = float(os.getenv("SIGLIP_SIMILARITY_THRESHOLD", "0.85"))
    siglip_crop_padding: float = float(os.getenv("SIGLIP_CROP_PADDING", "0.1"))

    # Re-ID parameters
    reid_max_age_frames: int = int(os.getenv("REID_MAX_AGE_FRAMES", "90"))  # 3s at 30fps
    reid_embedding_history: int = int(os.getenv("REID_EMBEDDING_HISTORY", "50"))

    # ==========================================================================
    # 10. API & WEBSOCKET STREAMING
    # ==========================================================================
    # FastAPI configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "1"))
    api_max_video_size_mb: int = int(os.getenv("API_MAX_VIDEO_SIZE_MB", "500"))
    api_timeout_seconds: int = int(os.getenv("API_TIMEOUT_SECONDS", "300"))
    api_cors_origins: str = os.getenv("API_CORS_ORIGINS", "*")

    # WebSocket streaming
    enable_websocket_streaming: bool = os.getenv("ENABLE_WEBSOCKET_STREAMING", "0") == "1"
    websocket_host: str = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    websocket_port: int = int(os.getenv("WEBSOCKET_PORT", "8765"))
    streaming_frame_skip: int = int(os.getenv("STREAMING_FRAME_SKIP", "1"))
    streaming_buffer_size: int = int(os.getenv("STREAMING_BUFFER_SIZE", "30"))
    streaming_quality: int = int(os.getenv("STREAMING_QUALITY", "80"))

    # Batch processing
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    max_concurrent_videos: int = int(os.getenv("MAX_CONCURRENT_VIDEOS", "2"))

    # Caching
    enable_model_caching: bool = os.getenv("ENABLE_MODEL_CACHING", "1") == "1"
    cache_dir: Path = Path(os.getenv("CACHE_DIR", "/tmp/bball_cv_cache"))
    embedding_cache_size: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

    # ==========================================================================
    # 11. VISUALIZATION & DEBUG
    # ==========================================================================
    # Court configuration
    court_config: CourtConfiguration = CourtConfiguration(
        league=League.NBA, measurement_unit=MeasurementUnit.FEET
    )
    court_scale: int = 20
    court_padding: int = 50
    court_line_thickness: int = 4

    # Colors
    palette: sv.ColorPalette = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
    ])
    magenta: sv.Color = sv.Color.from_hex("#FF1493")
    cyan: sv.Color = sv.Color.from_hex("#00BFFF")

    # Team colors
    team_a_color: sv.Color = sv.Color.from_hex("#1F77B4")  # blue
    team_b_color: sv.Color = sv.Color.from_hex("#FF7F0E")  # orange
    referee_color: sv.Color = sv.Color.from_hex("#808080")  # grey

    # Shot result colors
    attempt_color: sv.Color = sv.Color.from_hex("#1F77B4")
    made_color: sv.Color = sv.Color.from_hex("#007A33")  # green
    miss_color: sv.Color = sv.Color.from_hex("#850101")  # red

    # Debug options
    smoke_max_frames: Optional[int] = 200
    start_frame_index: int = int(os.getenv("START_FRAME_INDEX", "65"))
    save_debug_stage_images: bool = os.getenv("SAVE_DEBUG_STAGE_IMAGES", "1") == "1"
    enable_ffmpeg_compression: bool = os.getenv("ENABLE_FFMPEG_COMPRESSION", "0") == "1"
    save_event_frames: bool = os.getenv("SAVE_EVENT_FRAMES", "1") == "1"
    event_frame_limit: int = int(os.getenv("EVENT_FRAME_LIMIT", "12"))
    show_images: bool = os.getenv("SHOW_IMAGES", "0") == "1"
    strict_fail: bool = os.getenv("STRICT_FAIL", "1") == "1"
    emit_summary_image: bool = os.getenv("EMIT_SUMMARY_IMAGE", "1") == "1"
    preview_images_in_terminal: bool = os.getenv("PREVIEW_IMAGES_IN_TERMINAL", "0") == "1"
    terminal_preview_max_width: int = int(os.getenv("TERMINAL_PREVIEW_MAX_WIDTH", "80"))

    # ==========================================================================
    # 12. SHOT EVENT LOGIC
    # ==========================================================================
    reset_time_seconds: float = 1.7
    min_between_starts_seconds: float = 0.5
    cooldown_after_made_seconds: float = 0.5
    keypoint_smoothing_len: int = 3

    # ==========================================================================
    # 13. KINEMATICS STANDARDIZATION
    # ==========================================================================
    # Enable standardized output for SPL/mplbasketball compatibility
    enable_kinematics_export: bool = os.getenv("ENABLE_KINEMATICS_EXPORT", "0") == "1"

    # Court type for coordinate system
    kinematics_court_type: str = os.getenv("KINEMATICS_COURT_TYPE", "NBA")  # NBA, NCAA, FIBA

    # Output units
    kinematics_output_units: str = os.getenv("KINEMATICS_UNITS", "metres")  # metres, feet

    # Player height estimation
    default_player_height_m: float = float(os.getenv("DEFAULT_PLAYER_HEIGHT", "2.01"))  # ~6'7"
    estimate_z_from_ratios: bool = os.getenv("ESTIMATE_Z_FROM_RATIOS", "1") == "1"

    # SPL-Open-Data adapter settings
    spl_fps: float = float(os.getenv("SPL_FPS", "120.0"))
    spl_dataset_id: str = os.getenv("SPL_DATASET_ID", "SPL_FREETHROW")

    # Export paths
    kinematics_output_dir: Path = output_dir / "kinematics"
    kinematics_format: str = os.getenv("KINEMATICS_FORMAT", "parquet")  # parquet, csv


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def ensure_dirs(cfg: CVConfig) -> None:
    """Create output directories if they don't exist."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def resolve_api_key() -> str:
    """Return API key from env, supporting ROBOFLOW_API_KEY or INFERENCE_API_KEY."""
    api_key = os.getenv("ROBOFLOW_API_KEY") or os.getenv("INFERENCE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set ROBOFLOW_API_KEY or INFERENCE_API_KEY in the environment."
        )
    return api_key


def _set_inference_feature_flags() -> None:
    """
    Disable optional inference models we don't use to avoid warnings.
    Enable specific models based on config when needed.
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
    """
    Load core detection models (player + court).

    Args:
        cfg: CVConfig instance

    Returns:
        (player_model, court_model)
    """
    _set_inference_feature_flags()
    from inference import get_model

    api_key = resolve_api_key()
    player_model = get_model(model_id=cfg.player_model_id, api_key=api_key)
    court_model = get_model(model_id=cfg.court_model_id, api_key=api_key)
    return player_model, court_model


def load_ball_model(cfg: CVConfig):
    """Load ball detection model for shot arc analysis."""
    _set_inference_feature_flags()
    from inference import get_model

    api_key = resolve_api_key()
    return get_model(model_id=cfg.ball_model_id, api_key=api_key)


def load_jersey_model(cfg: CVConfig):
    """Load jersey number detection model."""
    _set_inference_feature_flags()
    from inference import get_model

    api_key = resolve_api_key()
    return get_model(model_id=cfg.jersey_number_model_id, api_key=api_key)


def court_base_image(cfg: CVConfig):
    """Generate base court diagram for visualization."""
    from sports.basketball import draw_court

    return draw_court(
        config=cfg.court_config,
        scale=cfg.court_scale,
        padding=cfg.court_padding,
        line_thickness=cfg.court_line_thickness,
    )


def get_enabled_features(cfg: CVConfig) -> Dict[str, bool]:
    """Get summary of enabled pipeline features."""
    return {
        "tracking": cfg.enable_tracking,
        "sam2_tracking": cfg.enable_sam2_tracking,
        "segment_homography": cfg.enable_segment_homography,
        "jersey_ocr": cfg.enable_jersey_ocr,
        "pose_estimation": cfg.enable_pose_estimation,
        "shot_arc_analysis": cfg.enable_shot_arc_analysis,
        "siglip_reid": cfg.enable_siglip_reid,
        "websocket_streaming": cfg.enable_websocket_streaming,
        "zone_detection": cfg.enable_zone_detection,
        "kinematics_export": cfg.enable_kinematics_export,
    }


def validate_config(cfg: CVConfig) -> None:
    """
    Validate config for required parameters when features are enabled.

    Fails fast with clear error messages instead of silent fallbacks.
    This should be called after loading config to catch misconfigurations early.

    Raises:
        ValueError: If required parameters are missing for enabled features
    """
    errors = []

    # Shot arc analysis requires ball model
    if cfg.enable_shot_arc_analysis and not cfg.ball_model_id:
        errors.append("Shot arc analysis enabled but BALL_MODEL_ID is empty")

    # SigLIP re-ID requires model name
    if cfg.enable_siglip_reid and not cfg.siglip_model_name:
        errors.append("SigLIP re-ID enabled but SIGLIP_MODEL_NAME is empty")

    # Jersey OCR with smolvlm2 requires model name
    if cfg.enable_jersey_ocr and cfg.jersey_ocr_type == "smolvlm2":
        if not cfg.smolvlm2_model_name:
            errors.append("SmolVLM2 jersey OCR enabled but SMOLVLM2_MODEL is empty")

    # Pose estimation requires model name
    if cfg.enable_pose_estimation and not cfg.pose_model_name:
        errors.append("Pose estimation enabled but POSE_MODEL_NAME is empty")

    # Segment homography with semantic constraints
    if cfg.enable_segment_homography and cfg.enable_semantic_constraints:
        if cfg.three_point_radius_ft <= 0:
            errors.append("Semantic constraints enabled but three_point_radius_ft invalid")

    # Kinematics export validation
    if cfg.enable_kinematics_export:
        if cfg.video_fps <= 0:
            errors.append(
                "Kinematics export enabled but VIDEO_FPS <= 0. "
                "Set VIDEO_FPS to a positive integer."
            )
        if cfg.kinematics_format not in {"parquet", "csv"}:
            errors.append(
                f"Unsupported KINEMATICS_FORMAT '{cfg.kinematics_format}'. "
                "Use 'parquet' or 'csv'."
            )

    if errors:
        raise ValueError("Config validation failed:\n- " + "\n- ".join(errors))


def describe_cv_config(cfg: CVConfig) -> str:
    """
    Generate human-readable summary of config for logging/debugging.

    Use this at pipeline startup to log the current configuration.

    Args:
        cfg: CVConfig instance

    Returns:
        Formatted string with key config info
    """
    features = get_enabled_features(cfg)
    enabled = [k for k, v in features.items() if v]
    disabled = [k for k, v in features.items() if not v]

    lines = [
        f"CV Pipeline Config (profile: {cfg.cv_profile})",
        f"  Enabled: {', '.join(enabled) if enabled else 'none'}",
        f"  Disabled: {', '.join(disabled) if disabled else 'none'}",
        f"  Models:",
        f"    Player: {cfg.player_model_id}",
        f"    Court: {cfg.court_model_id}",
        f"    Ball: {cfg.ball_model_id}",
        f"    Pose: {cfg.pose_model_name}",
    ]

    if cfg.enable_siglip_reid:
        lines.append(f"    SigLIP: {cfg.siglip_model_name}")

    if cfg.enable_jersey_ocr:
        lines.append(f"    Jersey OCR: {cfg.jersey_ocr_type}")

    lines.append(f"  FPS: {cfg.video_fps}")

    return "\n".join(lines)


def load_cv_config(validate: bool = True) -> CVConfig:
    """
    Load CVConfig from environment with profile application and validation.

    This is the recommended way to load config in all pipeline modules.
    It applies profile presets first, then validates the configuration.

    Args:
        validate: Whether to run validation (default True)

    Returns:
        Configured CVConfig instance

    Raises:
        ValueError: If profile is unknown or validation fails

    Usage:
        cfg = load_cv_config()
        print(describe_cv_config(cfg))
    """
    # Create base config from env
    cfg = CVConfig()

    # Note: Since CVConfig is frozen, we can't modify it after creation.
    # The profile system works by having users set the corresponding env vars.
    # This function primarily validates and describes the config.

    # Check profile is valid
    if cfg.cv_profile not in PROFILE_PRESETS and cfg.cv_profile != "custom":
        valid = ", ".join(PROFILE_PRESETS.keys())
        raise ValueError(f"Unknown CV_PROFILE '{cfg.cv_profile}'. Valid options: {valid}, custom")

    # Validate if requested
    if validate:
        validate_config(cfg)

    return cfg


def apply_profile_to_env(profile: str) -> None:
    """
    Apply a profile's settings to environment variables.

    Call this BEFORE creating CVConfig to have profile settings take effect.
    Useful for programmatic profile switching.

    Args:
        profile: Profile name from PROFILE_PRESETS

    Example:
        apply_profile_to_env("full_biomech")
        cfg = load_cv_config()
    """
    if profile not in PROFILE_PRESETS:
        valid = ", ".join(PROFILE_PRESETS.keys())
        raise ValueError(f"Unknown profile '{profile}'. Valid options: {valid}")

    preset = PROFILE_PRESETS[profile]

    # Map config field names to env var names
    field_to_env = {
        "enable_tracking": "ENABLE_TRACKING",
        "enable_segment_homography": "ENABLE_SEGMENT_HOMOGRAPHY",
        "enable_pose_estimation": "ENABLE_POSE_ESTIMATION",
        "enable_shot_arc_analysis": "ENABLE_SHOT_ARC_ANALYSIS",
        "enable_siglip_reid": "ENABLE_SIGLIP_REID",
        "enable_jersey_ocr": "ENABLE_JERSEY_OCR",
        "enable_websocket_streaming": "ENABLE_WEBSOCKET_STREAMING",
        "enable_kinematics_export": "ENABLE_KINEMATICS_EXPORT",
        "enable_shot_form_analysis": "ENABLE_SHOT_FORM",
        "streaming_frame_skip": "STREAMING_FRAME_SKIP",
    }

    for field, value in preset.items():
        env_var = field_to_env.get(field)
        if env_var:
            if isinstance(value, bool):
                os.environ[env_var] = "1" if value else "0"
            else:
                os.environ[env_var] = str(value)

    os.environ["CV_PROFILE"] = profile
