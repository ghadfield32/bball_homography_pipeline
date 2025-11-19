# api/src/cv/api_endpoints.py
"""
FastAPI endpoints for CV pipeline access.

Exposes basketball shot detection, tracking, and analysis via REST API.

Usage:
    # Start server
    uvicorn api.src.cv.api_endpoints:app --host 0.0.0.0 --port 8000

    # Or with config
    python -m api.src.cv.api_endpoints
"""
from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api.src.cv.config import CVConfig


# --- Pydantic models for API ---

class ProcessingConfig(BaseModel):
    """Configuration for video processing."""
    enable_tracking: bool = True
    enable_pose_estimation: bool = False
    enable_shot_arc_analysis: bool = False
    enable_jersey_ocr: bool = False
    confidence_threshold: float = 0.3
    max_frames: Optional[int] = None


class ShotEvent(BaseModel):
    """Single shot event."""
    shot_id: int
    start_frame: int
    end_frame: int
    result: str  # "made" or "miss"
    shooter_track_id: Optional[int] = None
    shot_type: str = "unknown"
    court_position: Optional[List[float]] = None


class TrackInfo(BaseModel):
    """Tracking information for a player."""
    track_id: int
    team_id: int
    total_distance_ft: float
    max_speed_fps: float
    jersey_number: Optional[str] = None


class ProcessingResult(BaseModel):
    """Result of video processing."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    shots: List[ShotEvent] = []
    tracks: List[TrackInfo] = []
    total_frames: int = 0
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
    gpu_available: bool


# --- App initialization ---

app = FastAPI(
    title="Basketball CV Pipeline API",
    description="REST API for basketball shot detection, tracking, and analysis",
    version="0.3.0",
)

# CORS middleware
cfg = CVConfig()
origins = cfg.api_cors_origins.split(",") if cfg.api_cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (use Redis/DB for production)
_jobs: Dict[str, ProcessingResult] = {}
_models_loaded = False


# --- Helper functions ---

def _check_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


async def _process_video_async(
    job_id: str,
    video_path: Path,
    config: ProcessingConfig,
) -> None:
    """Process video asynchronously."""
    global _jobs

    start_time = time.time()
    _jobs[job_id].status = "processing"

    try:
        # Import here to avoid slow startup
        from api.src.cv.shot_pipeline import process_video_enhanced
        from api.src.cv.config import CVConfig

        # Create config with overrides
        cfg = CVConfig()

        # Run processing in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _run_pipeline(video_path, cfg, config, job_id)
        )

        # Update job with results
        _jobs[job_id].shots = result.get("shots", [])
        _jobs[job_id].tracks = result.get("tracks", [])
        _jobs[job_id].total_frames = result.get("total_frames", 0)
        _jobs[job_id].status = "completed"
        _jobs[job_id].progress = 1.0

    except Exception as e:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)

    finally:
        _jobs[job_id].processing_time_seconds = time.time() - start_time


def _run_pipeline(
    video_path: Path,
    cfg: CVConfig,
    proc_config: ProcessingConfig,
    job_id: str,
) -> Dict[str, Any]:
    """Run the CV pipeline synchronously."""
    import cv2
    from api.src.cv.shot_pipeline import process_video_with_state

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Process video
    result = process_video_with_state(
        video_path=video_path,
        cfg=cfg,
        progress_callback=lambda p: _update_progress(job_id, p),
    )

    # Convert to API format
    shots = []
    for i, shot in enumerate(result.get("shot_events", [])):
        shots.append(ShotEvent(
            shot_id=i,
            start_frame=shot.get("start_frame", 0),
            end_frame=shot.get("end_frame", 0),
            result=shot.get("result", "unknown"),
            shooter_track_id=shot.get("shooter_track_id"),
            shot_type=shot.get("shot_type", "unknown"),
            court_position=shot.get("court_position"),
        ))

    # Convert tracking info
    tracks = []
    for track_id, track_data in result.get("tracking_analytics", {}).get("tracks", {}).items():
        tracks.append(TrackInfo(
            track_id=int(track_id),
            team_id=track_data.get("team_id", -1),
            total_distance_ft=track_data.get("total_distance_ft", 0.0),
            max_speed_fps=track_data.get("max_speed_fps", 0.0),
            jersey_number=track_data.get("jersey_number"),
        ))

    return {
        "shots": shots,
        "tracks": tracks,
        "total_frames": total_frames,
    }


def _update_progress(job_id: str, progress: float) -> None:
    """Update job progress."""
    if job_id in _jobs:
        _jobs[job_id].progress = progress


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.3.0",
        models_loaded=_models_loaded,
        gpu_available=_check_gpu(),
    )


@app.post("/process", response_model=ProcessingResult)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config: Optional[ProcessingConfig] = None,
):
    """
    Upload and process a video for shot detection.

    Returns a job ID for tracking progress.
    """
    if config is None:
        config = ProcessingConfig()

    # Validate file
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file size
    cfg = CVConfig()
    max_size = cfg.api_max_video_size_mb * 1024 * 1024

    # Save to temp file
    job_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / "bball_cv"
    temp_dir.mkdir(exist_ok=True)
    video_path = temp_dir / f"{job_id}_{video.filename}"

    try:
        content = await video.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Video exceeds max size of {cfg.api_max_video_size_mb}MB"
            )

        with open(video_path, "wb") as f:
            f.write(content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Create job
    _jobs[job_id] = ProcessingResult(
        job_id=job_id,
        status="pending",
        progress=0.0,
    )

    # Start processing in background
    background_tasks.add_task(_process_video_async, job_id, video_path, config)

    return _jobs[job_id]


@app.get("/jobs/{job_id}", response_model=ProcessingResult)
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _jobs[job_id]


@app.get("/jobs/{job_id}/shots", response_model=List[ShotEvent])
async def get_job_shots(job_id: str):
    """Get shot events for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job.status}")

    return job.shots


@app.get("/jobs/{job_id}/tracks", response_model=List[TrackInfo])
async def get_job_tracks(job_id: str):
    """Get tracking info for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job.status}")

    return job.tracks


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del _jobs[job_id]

    # Clean up temp files
    temp_dir = Path(tempfile.gettempdir()) / "bball_cv"
    for f in temp_dir.glob(f"{job_id}_*"):
        try:
            f.unlink()
        except Exception:
            pass

    return {"status": "deleted"}


@app.post("/analyze/frame")
async def analyze_single_frame(
    image: UploadFile = File(...),
):
    """
    Analyze a single frame for players and court.

    Returns detections and homography if available.
    """
    from api.src.cv.config import CVConfig, load_models
    from api.src.cv.shot_pipeline import detect_players, detect_court
    import cv2

    # Read image
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    cfg = CVConfig()
    player_model, court_model = load_models(cfg)

    # Detect players
    player_dets = detect_players(frame, player_model, cfg)

    # Detect court
    court_dets = detect_court(frame, court_model, cfg)

    # Format response
    players = []
    if player_dets is not None:
        for i in range(len(player_dets)):
            players.append({
                "bbox": player_dets.xyxy[i].tolist(),
                "confidence": float(player_dets.confidence[i]) if player_dets.confidence is not None else 1.0,
            })

    keypoints = []
    if court_dets is not None and hasattr(court_dets, "xy") and court_dets.xy is not None:
        for i in range(len(court_dets.xy)):
            keypoints.append({
                "position": court_dets.xy[i].tolist(),
                "class_id": int(court_dets.class_id[i]) if court_dets.class_id is not None else 0,
            })

    return {
        "players": players,
        "court_keypoints": keypoints,
        "frame_shape": list(frame.shape),
    }


@app.get("/config")
async def get_config():
    """Get current configuration parameters."""
    cfg = CVConfig()

    # Return serializable config
    return {
        "tracking": {
            "enabled": cfg.enable_tracking,
            "activation_threshold": cfg.track_activation_threshold,
            "lost_track_buffer": cfg.lost_track_buffer,
        },
        "pose": {
            "enabled": cfg.enable_pose_estimation,
            "model": cfg.pose_model_name,
        },
        "homography": {
            "enabled": cfg.enable_segment_homography,
            "rmse_max_court": cfg.homography_rmse_court_max,
            "rmse_max_image": cfg.homography_rmse_image_max,
        },
        "models": {
            "player": cfg.player_model_id,
            "court": cfg.court_model_id,
            "ball": cfg.ball_model_id,
        },
        "thresholds": {
            "confidence": cfg.confidence_threshold,
            "iou": cfg.iou_threshold,
        },
    }


# --- Main entry point ---

def main():
    """Run the API server."""
    import uvicorn

    cfg = CVConfig()
    uvicorn.run(
        "api.src.cv.api_endpoints:app",
        host=cfg.api_host,
        port=cfg.api_port,
        workers=cfg.api_workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
