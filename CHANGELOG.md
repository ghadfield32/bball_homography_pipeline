# Project Changelog: Basketball Homography Pipeline

**Last Updated**: 2025-11-19
**Purpose**: Track changes, optimizations, and development tasks to prevent duplication and ensure efficient iteration

---

## [Current] CV Pipeline Enhancements Phase 3 - API, Streaming, Shot Arc, Re-ID - 2025-11-19

### 🎯 Objective
Add production-ready API endpoints, real-time streaming, shot arc analysis, and visual re-identification for complete pipeline exposure.

### ✅ Comprehensive Config Parameters

**Modified File**: `api/src/cv/config.py` (+60 lines)

Added configuration sections for all Phase 3 features:
- Jersey OCR: `enable_jersey_ocr`, `jersey_ocr_type`, `jersey_ocr_confidence`
- Shot Arc Analysis: `ball_model_id`, `arc_min_trajectory_points`, `arc_velocity_window_frames`
- SigLIP Re-ID: `siglip_model_name`, `siglip_similarity_threshold`, `siglip_embedding_dim`
- WebSocket Streaming: `websocket_host`, `websocket_port`, `streaming_frame_skip`
- FastAPI: `api_host`, `api_port`, `api_workers`, `api_max_video_size_mb`
- Batch Processing: `batch_size`, `max_concurrent_videos`
- Caching: `enable_model_caching`, `cache_dir`, `embedding_cache_size`

### ✅ FastAPI CV Endpoints

**New File**: `api/src/cv/api_endpoints.py` (~400 lines)

REST API for pipeline access:
- `POST /process` - Upload video for async processing
- `GET /jobs/{job_id}` - Get job status and progress
- `GET /jobs/{job_id}/shots` - Get shot events for completed job
- `GET /jobs/{job_id}/tracks` - Get tracking info
- `POST /analyze/frame` - Analyze single frame
- `GET /config` - Get current configuration
- `GET /health` - Health check

Usage:
```bash
uvicorn api.src.cv.api_endpoints:app --host 0.0.0.0 --port 8000
```

### ✅ Shot Arc Analysis Module

**New File**: `api/src/cv/shot_arc.py` (~450 lines)

Ball trajectory analysis:
- `ShotArcAnalyzer` class with ball detection and tracking
- `ArcMetrics` dataclass with release angle, entry angle, apex height, velocity
- Parabolic curve fitting with R-squared validation
- Trajectory smoothing and velocity computation
- Scale estimation from ball size

Key metrics computed:
- Release angle (degrees)
- Entry angle (degrees)
- Apex height (pixels/feet)
- Release velocity (pixels/frame or ft/s)

Usage:
```python
from api.src.cv.shot_arc import create_shot_arc_analyzer

analyzer = create_shot_arc_analyzer(cfg)
metrics = analyzer.analyze_shot_arc(start_frame, end_frame, fps=30)
print(f"Release angle: {metrics.release_angle}°")
```

### ✅ SigLIP Visual Re-Identification

**New File**: `api/src/cv/siglip_reid.py` (~400 lines)

Appearance-based re-identification:
- `SigLIPReID` class using HuggingFace SigLIP model
- `EmbeddingHistory` with temporal weighting
- Cosine similarity matching
- Track merging across camera cuts

Features:
- Embedding extraction from player crops
- Similarity matrix computation
- Lost track detection and matching
- Configurable similarity threshold

Usage:
```python
from api.src.cv.siglip_reid import create_siglip_reid

reid = create_siglip_reid(cfg)
reid.update(frame, tracked_dets, frame_idx)
matches = reid.attempt_reidentification(new_track_ids, frame_idx)
```

### ✅ WebSocket Streaming

**New File**: `api/src/cv/websocket_stream.py` (~350 lines)

Real-time video processing:
- `WebSocketStreamServer` class
- Per-session state management
- Frame-by-frame processing with immediate results
- JSON command protocol (ping, reset, stats)

Usage:
```bash
python -m api.src.cv.websocket_stream
# Server starts on ws://localhost:8765
```

Client example (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.send(frameData);  // Send JPEG frame
ws.onmessage = (e) => {
    const result = JSON.parse(e.data);
    // result.players, result.shot_event, etc.
};
```

### 📊 Files Changed Summary

| File | Action | Lines |
|------|--------|-------|
| `api/src/cv/config.py` | MODIFIED | +60 |
| `api/src/cv/api_endpoints.py` | NEW | ~400 |
| `api/src/cv/shot_arc.py` | NEW | ~450 |
| `api/src/cv/siglip_reid.py` | NEW | ~400 |
| `api/src/cv/websocket_stream.py` | NEW | ~350 |

### ⚠️ New Dependencies

Optional dependencies for Phase 3 features:
```bash
# For SigLIP re-ID
pip install transformers torch

# For WebSocket streaming
pip install websockets

# Already included: fastapi, uvicorn
```

---

## [Previous] CV Pipeline Enhancements Phase 2 - Tuning, Constraints, OCR, Testing - 2025-11-19

### 🎯 Objective
Enhance pipeline with basketball-specific tuning, semantic validation, jersey OCR, and test utilities.

### ✅ ByteTrack Parameter Tuning for Basketball

**Modified Files**: `api/src/cv/tracker.py`, `api/src/cv/config.py`

Tuned defaults for basketball-specific tracking challenges:
- `track_activation_threshold`: 0.25 → 0.20 (catch partially occluded players)
- `lost_track_buffer`: 30 → 60 frames (~2s for drives/screens)
- `minimum_matching_threshold`: 0.8 → 0.6 (accommodate fast-moving players)
- `minimum_consecutive_frames`: 1 → 2 (reduce flickering)

### ✅ Semantic Constraints for Homography

**Modified File**: `api/src/cv/homography_calibrator.py` (+140 lines)

Added validation methods:
- `_validate_line_collinearity()`: Ensures baseline/sideline points remain collinear
- `_validate_arc_radius()`: Validates 3-point arc radius (~23.75ft)
- `validate_semantic_constraints()`: Main validation orchestrator

New configuration:
```python
enable_semantic_constraints: bool = True
line_collinearity_threshold: float = 0.5  # feet
arc_radius_threshold: float = 1.0  # feet
three_point_radius_ft: float = 23.75  # NBA standard
```

### ✅ Jersey Number OCR Module

**New File**: `api/src/cv/jersey_ocr.py` (~350 lines)

Features:
- `JerseyOCR` class with pluggable OCR backends (EasyOCR, PaddleOCR)
- `TrackNumberHistory` with majority voting for stable number assignments
- Per-track number persistence and confidence scoring
- Track merging for re-identification across camera cuts
- Number region extraction from player bounding boxes

Usage:
```python
from api.src.cv.jersey_ocr import JerseyOCR

ocr = JerseyOCR()
ocr.load_model("easyocr")  # or "paddleocr"

numbers = ocr.detect_and_update(frame, tracked_dets)
player_number = ocr.get_track_number(track_id)
```

### ✅ Test Utilities

**New File**: `api/src/cv/test_pipeline.py` (~400 lines)

Comprehensive test suite:
- `test_tracker_module()`: PlayerTracker, TrackState, analytics
- `test_homography_calibrator()`: Calibrator, SegmentData, semantic validation
- `test_pose_pipeline()`: PosePipeline, PoseObservation, PlayerPoseHistory
- `test_jersey_ocr()`: JerseyOCR, number history, track merging
- `test_config()`: Verify all new config parameters

Run tests:
```bash
python -m api.src.cv.test_pipeline
python -m api.src.cv.test_pipeline --component tracker
```

### 📊 Files Changed Summary

| File | Action | Lines |
|------|--------|-------|
| `api/src/cv/tracker.py` | MODIFIED | +15 |
| `api/src/cv/config.py` | MODIFIED | +5 |
| `api/src/cv/homography_calibrator.py` | MODIFIED | +140 |
| `api/src/cv/jersey_ocr.py` | NEW | ~350 |
| `api/src/cv/test_pipeline.py` | NEW | ~400 |

---

## [Previous] CV Pipeline Enhancements Phase 1 - Tracking, Homography, Pose - 2025-11-19

### 🎯 Objective
Add production-grade enhancements to the CV pipeline: persistent player tracking, segment-level homography, and pose estimation for biomechanics analysis.

### ✅ Phase 1: ByteTrack Player Tracking Module

**New File**: `api/src/cv/tracker.py` (~300 lines)

Key features:
- `PlayerTracker` class using supervision's ByteTrack for multi-object tracking
- `TrackState` dataclass with history, team voting, and analytics (speed, distance)
- Persistent track IDs across frames with configurable parameters
- Team label attachment using majority voting for stability
- Integration helper `assign_teams_to_detections()` for existing pipeline

Configuration (in `config.py`):
```python
enable_tracking: bool = True
track_activation_threshold: float = 0.25
lost_track_buffer: int = 30
minimum_matching_threshold: float = 0.8
```

### ✅ Phase 2: Segment-Level Homography Calibrator

**New File**: `api/src/cv/homography_calibrator.py` (~400 lines)

Key features:
- `HomographyCalibrator` class for multi-frame H optimization
- Camera segment detection via keypoint centroid jumps
- Aggregates keypoints across segment frames for robust fit
- RANSAC + weighted least-squares refinement
- Quality metrics (RMSE court/image) per segment
- Interpolation for unreliable frames
- `get_quality_mask()` for downstream confidence-aware analytics

Configuration:
```python
enable_segment_homography: bool = False  # Off by default
segment_min_frames: int = 10
segment_change_threshold: float = 50.0  # pixels
```

### ✅ Phase 3: Pose Estimation Pipeline

**New File**: `api/src/cv/pose_pipeline.py` (~450 lines)

Key features:
- `PosePipeline` class using YOLO-pose models (yolov8n-pose, yolov8s-pose, yolov8m-pose)
- `PoseObservation` and `PlayerPoseHistory` for per-track pose data
- 17-keypoint COCO skeleton with court coordinate transformation
- Joint trajectory extraction for any keypoint
- `get_release_point_estimate()` for shot analysis
- `analyze_shot_form()` for elbow angle and joint sequences

Configuration:
```python
enable_pose_estimation: bool = False  # Off by default
pose_model_name: str = "yolov8n-pose"
video_fps: int = 30
```

### ✅ Phase 4: Enhanced Pipeline Integration

**Modified File**: `api/src/cv/shot_pipeline.py` (+280 lines)

New function `process_video_enhanced()` at line 2683:
- Integrates all three new modules with existing pipeline
- Team classification attached to tracks (stable labels)
- Returns extended metrics: tracking_analytics, homography_segments, pose_tracks
- Backward compatible - original `process_video()` unchanged

**Modified File**: `api/src/cv/config.py` (+20 lines)

Added configuration sections for:
- Tracking parameters
- Segment-level homography
- Pose estimation
- Video frame rate

### 📊 Files Changed Summary

| File | Action | Lines |
|------|--------|-------|
| `api/src/cv/tracker.py` | NEW | ~300 |
| `api/src/cv/homography_calibrator.py` | NEW | ~400 |
| `api/src/cv/pose_pipeline.py` | NEW | ~450 |
| `api/src/cv/shot_pipeline.py` | MODIFIED | +280 |
| `api/src/cv/config.py` | MODIFIED | +20 |

### 🔧 Usage

```python
from api.src.cv.config import CVConfig
from api.src.cv.shot_pipeline import process_video_enhanced

cfg = CVConfig()
cfg.enable_tracking = True
cfg.enable_segment_homography = True
cfg.enable_pose_estimation = True

results = process_video_enhanced(
    video_path=video_path,
    cfg=cfg,
    player_model=player_model,
    court_model=court_model,
)

# Extended results include:
# - tracking_analytics: {total_tracks, active_tracks, team_counts, avg_distance_ft, max_speed_fps}
# - homography_segments: number of camera segments detected
# - segment_quality: [{id, rmse_court, inliers}, ...]
# - pose_tracks: number of players with pose data
```

### ⚠️ Dependencies
No new dependencies - uses existing:
- `supervision>=0.26` (ByteTrack built-in)
- `ultralytics==8.3.158` (YOLO-pose models)

### 📋 Next Steps
- Test with sample videos to validate tracking accuracy
- Tune ByteTrack parameters for basketball occlusions
- Add jersey number OCR for player re-identification
- Implement semantic constraints for homography (line-on-line, arc radius)

---

## [Previous] CV Pipeline Function Missing Fix - 2025-11-19

### 🎯 Objective
Fix `NameError: name 'finalize_smoke_outputs' is not defined` in `piotr_automated_pipeline copy.ipynb`

### 🔬 Root Cause Analysis
- **Error**: `NameError` at line 3225 when calling `finalize_smoke_outputs()`
- **Cause**: Function was missing from notebook - gap between `compress_video` (line 3309) and `_file_ok` (line 3315) had only blank lines
- **Origin**: Function was either not copied when notebook was duplicated, or accidentally deleted during editing
- **Impact**: Smoke test pipeline could not complete after `process_video()` execution

### ✅ Fix Applied
Inserted missing functions into [piotr_automated_pipeline copy.ipynb](notebooks/backend/data_engineering/cv/piotr_automated_pipeline%20copy.ipynb):

1. **`compress_video()`** (line ~3288-3322): Compresses video files using ffmpeg with configurable CRF
2. **`finalize_smoke_outputs()`** (line ~3323-3366): Writes JSON manifest with all outputs, handles optional compression

### 📋 Functions Changed (Full Replacement)
```python
def compress_video(in_path: Path, out_path: Path, crf: int = 28) -> bool:
    if shutil.which("ffmpeg") is None:
        print("[DEBUG] ffmpeg not found; skipping compression.")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-vcodec", "libx264", "-crf", str(crf), str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[DEBUG] ffmpeg failed for {in_path.name}: {e}")
        return False


def finalize_smoke_outputs(
    cfg: CVConfig, metrics: Dict, stage_images: List[Path]
) -> Path:
    """Write a JSON manifest with all outputs (videos, optional compressed, stage images, events)."""
    overlay_path = Path(metrics["overlay_out"])
    court_path = Path(metrics["court_out"])

    overlay_compressed = None
    court_compressed = None

    if cfg.enable_ffmpeg_compression:
        overlay_compressed = overlay_path.parent / f"{overlay_path.stem}-compressed{overlay_path.suffix}"
        court_compressed = court_path.parent / f"{court_path.stem}-compressed{court_path.suffix}"
        compress_video(overlay_path, overlay_compressed, crf=28)
        compress_video(court_path, court_compressed, crf=28)

    manifest = {
        "video": str(metrics["video"]),
        "overlay_out": str(overlay_path),
        "court_out": str(court_path),
        "final_court_image": metrics.get("final_court_image"),
        "overlay_compressed": (str(overlay_compressed) if overlay_compressed else None),
        "court_compressed": (str(court_compressed) if court_compressed else None),
        "shots_total": metrics["shots_total"],
        "shots_made": metrics["shots_made"],
        "shots_missed": metrics["shots_missed"],
        "frames": metrics["frames"],
        "stage_images": [str(p) for p in stage_images if p is not None],
        "event_images": metrics.get("event_images", []),
    }

    manifest_path = overlay_path.parent / f"{overlay_path.stem}-smoke_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SMOKE] Wrote manifest: {manifest_path}")
    return manifest_path
```

### 📊 Verification
- ✅ All 53 functions present in notebook cell
- ✅ Function order correct: `compress_video` (29) → `finalize_smoke_outputs` (30) → `_file_ok` (31)
- ✅ Dependencies verified: `shutil`, `subprocess`, `json`, `Path` available

### ⚠️ Dependencies
Functions require these imports (should already be at notebook top):
```python
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, List
```

### 📋 Test the Fix
Re-run the notebook cell containing `finalize_smoke_outputs()` call:
```python
manifest_path = finalize_smoke_outputs(
    cfg=cfg,
    metrics=metrics,
    stage_images=[p for p in stage_images if p is not None],
)
```

### 🔧 Prevention
- Always compare notebook copies against originals when debugging
- Use `%%writefile` magic to export notebook code to .py for version control
- Reference canonical implementation in [shot_pipeline.py:1325](api/src/cv/shot_pipeline.py#L1325)

---

## [Previous] Node.js 20 Installation for Claude Code - 2025-11-18

### 🎯 Objective
Fix Claude Code extension error: "Claude Code requires Node.js version 18 or higher to be installed"

### ✅ Fix Applied

#### Added Node.js 20 LTS to Dockerfile ([Dockerfile:59-66](.devcontainer/Dockerfile#L59))
```dockerfile
# Install Node.js 20 LTS (required for Claude Code extension)
RUN echo "STEP: Installing Node.js 20 LTS..." && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest && \
    node --version && npm --version
```

Also added `gnupg` to system dependencies for Node.js installation.

### 📋 Rebuild Required
This change requires rebuilding the Docker image:
```bash
cd .devcontainer
docker-compose build datascience
docker-compose up -d
```

---

## [Previous] GitHub Dependencies Fix - Build Failure - 2025-11-18

### 🎯 Objective
Fix Docker build failures caused by GitHub HTTP 500 errors when fetching git-based dependencies

### ✅ BUILD STATUS: IMAGE BUILT SUCCESSFULLY
**Previous Errors Fixed**:
1. `failed to download pytube @ git+https://github.com/pytube/pytube@a32fff39...`
2. `failed to download sports @ git+https://github.com/roboflow/sports.git@bb203feb...`

**Build Result**: All 30/30 layers completed, image created in 1416.7s
**New Issue**: 502 Bad Gateway when starting containers (Docker Desktop engine overloaded)

**Root Cause**: Git-based dependencies in pyproject.toml were unreliable; Windows mount I/O caused Docker Desktop exhaustion

### ✅ Fixes Applied

#### 1. pytube: Changed from git to PyPI ([pyproject.toml:89](pyproject.toml#L89))
```toml
# Before:
"pytube @ git+https://github.com/pytube/pytube",

# After:
"pytube>=15.0.0",  # YouTube download library
```

#### 2. sports: Moved to postCreateCommand ([pyproject.toml:108-111](pyproject.toml#L108), [devcontainer.json:113](devcontainer.json#L113))
- Removed from pyproject.toml dependencies (not on PyPI, GitHub-only)
- Added to postCreateCommand with graceful fallback
- Will install after container starts when GitHub becomes available

```toml
# pyproject.toml - Commented out:
# "sports @ git+https://github.com/roboflow/sports.git@bb203feb..."
```

```json
// devcontainer.json postCreateCommand now includes:
(uv pip install git+https://github.com/roboflow/sports.git@bb203feb... && echo "✅ Sports installed") || echo "⚠️ Install manually later"
```

#### 3. Deleted uv.lock
- Removed stale lock file with GitHub commit references
- Will regenerate on next build

### 📋 Why This Works
- **pytube**: PyPI releases are stable and always available
- **sports**: postCreateCommand allows build to succeed; package installs later
- No blocking dependency on GitHub server availability
- Build can complete even if sports installation fails

### 🔧 502 Bad Gateway Resolution

The 502 error occurred because Docker Desktop became overwhelmed during the 381.7s image export. **The image was built successfully** - just need to restart Docker Desktop:

```powershell
# 1. Restart Docker Desktop (use system tray or Task Manager)
# 2. Wait for Docker to be fully ready
# 3. Start containers:
cd .devcontainer
docker-compose up -d
docker-compose ps
```

### ⚠️ Notes
- If sports fails to install in postCreateCommand, run manually:
  ```bash
  uv pip install git+https://github.com/roboflow/sports.git
  ```
- yt-dlp (already in project) is more actively maintained than pytube
- Consider migrating pytube usage to yt-dlp in the future
- **IMPORTANT**: Move project to WSL filesystem for reliable builds (see previous changelog entries)

---

## [Previous] ⚠️ CRITICAL: unpigz Deflate Corruption - Build Failure - 2025-11-18

### 🎯 Objective
Debug Docker build failures: Progression from EOF errors → tar corruption → **unpigz deflate corruption**

### ❌ BUILD STATUS: BLOCKING FAILURE (Build 4)
**Status**: All 30/30 layers built, but **LAYER EXTRACTION FAILS** with gzip corruption
**Error**: `unpigz: skipping: <stdin>: corrupted -- invalid deflate data (invalid literal/lengths set)`
**Layer**: `sha256:b7ad595cf7807a869f80df56107214b51b9767d66e1833f6733d6e11add603e7`
**Export Time**: 277.5s (2.8x normal, progressively worse each build)
**Cause**: Windows mount 9P protocol bit-level corruption during prolonged gzip writes
**Action Required**: **MUST move project to WSL filesystem OR nuclear Docker reset**

### 📊 Corruption Progression Analysis

| Build | Export Time | Error Type | Corruption Level | Image Created |
|-------|-------------|------------|------------------|---------------|
| 2 | 85.9s | EOF (cosmetic) | None | ✅ Yes |
| 3 | 241.5s | tar header | Structural (header bytes) | ❌ No |
| 4 | 277.5s | **unpigz deflate** | **Byte-level (gzip stream)** | ❌ No |

**Pattern**: Export duration increasing ~30% per build, corruption depth increasing from structural to bit-level

### 🔬 Technical Analysis: unpigz vs tar Corruption

**Previous (Build 3)**: `archive/tar: invalid tar header`
- Tar headers are fixed 512-byte structures
- Header checksum failed → structural corruption
- Docker could identify the corrupt point

**Current (Build 4)**: `unpigz: corrupted -- invalid deflate data`
- Gzip uses deflate algorithm with Huffman coding
- A single bit flip corrupts the entire deflate stream
- unpigz cannot decompress → layer extraction fails
- **Deeper corruption** - data is corrupt before tar extraction even begins

### 🔍 Root Cause: 9P Protocol Bit-Level Corruption

1. Docker creates 18.1GB image with compressed layers
2. Export writes to Docker storage through WSL2 9P filesystem bridge
3. **277.5 seconds** of sustained I/O causes protocol buffer issues
4. WSL2 9P translation corrupts gzip deflate stream at bit level
5. unpigz detects invalid Huffman codes in deflate data
6. Layer extraction fails completely
7. Container cannot start

**Why It's Getting Worse**: Each build slightly increases export time (cache state, layer sizes), increasing corruption probability

### 📊 Failure Progression

| Build | Context | Layers | Export | Image | Status |
|-------|---------|--------|--------|-------|--------|
| Build 1 | 78s | Timeout @ 25/30 | N/A | ❌ | Failed - timeout |
| Build 2 | 1.2s | ✅ 30/30 | 85.9s EOF | ✅ Created | Success (cosmetic EOF) |
| Build 3 | 1.0s | ✅ 30/30 | **241.5s TAR CORRUPTION** | ❌ **FAILED** | **BLOCKING** |

**Pattern**: Export getting slower and now causing **tar header corruption** → image creation failure

### 🔍 Root Cause Analysis

**Primary Issue: Windows Mount Point I/O Bottleneck**
- Building from `/c/Users/...` (Windows NTFS mount in WSL2)
- **1.06MB context taking 78 seconds** to transfer (should be <1s)
- **50x slower** than native WSL2 filesystem
- Causes BuildKit connection timeout → `EOF error`

**Secondary Issue: Corrupted BuildKit Cache**
- 15.84GB build cache from previous failed builds
- Cache corruption from interrupted builds on slow filesystem
- Stale layers causing build inconsistencies

**Tertiary Issue: Large Build Context**
- .dockerignore only had 23 lines, missing major exclusions
- 27MB `notebooks/` directory being sent to daemon
- Markdown files, scripts, and documentation unnecessarily included

**Minor Issue: HOME Variable**
- `${HOME}` not set in environment → warning in docker-compose
- X11 authority mount failing silently

### ✅ Fixes Implemented

#### 1. Enhanced .dockerignore (23 → 110 lines)
- **Added**: `notebooks/`, `*.ipynb`, `mlruns/`, `mlflow_db/`, `api/`, `scripts/`, `*.md`
- **Added**: Python artifacts exclusions (build/, dist/, *.egg-info/, etc.)
- **Added**: OS-specific files (.DS_Store, Thumbs.db)
- **Added**: Development tools (.claude/, .vscode/)
- **Result**: Build context reduced from ~28MB to ~1-2MB

#### 2. Fixed HOME Variable in docker-compose.yml:174
```yaml
# Before:
- ${HOME}/.Xauthority:/root/.Xauthority:rw

# After:
- ${HOME:-/root}/.Xauthority:/root/.Xauthority:rw
```
- Added default fallback to prevent undefined variable warnings
- X11 authentication now works even if HOME is unset

#### 3. Created Diagnostic & Fix Scripts
- **`diagnose_build_context.sh`**: Analyzes filesystem, context size, and cache
- **`fix_build_performance.sh`**: Automated fixes with --move-to-wsl, --clean-docker, --test-build
- **`quick_fix_build.sh`**: Immediate cache cleanup without moving files

### 🔧 Detailed Error Analysis

**Build Timeline (Failed Build):**
```
0-78s:     Context transfer stuck (1.06MB taking 78s)
78-1127s:  Layers 1-25/30 building (many cached)
1127s:     Layer 25/30 (CV validation) - connection lost
Error:     "failed to solve: Unavailable: error reading from server: EOF"
```

**Why EOF Occurred:**
1. Slow I/O from Windows mount kept connection open for 19 minutes
2. BuildKit daemon has timeout for idle/slow connections
3. After installing CV packages (step 24), validation step (25) triggered timeout
4. TCP/IPC connection to Docker daemon closed → `EOF`

**Cascading Failures:**
1. Build fails → corrupted layer cache
2. docker-compose tries to pull MLflow → fails with "unexpected end of JSON input"
3. Docker daemon state partially corrupted from interrupted build

### 📋 Resolution Steps

**Immediate Fixes (Without Moving Project):**
```bash
# 1. Clean corrupted cache
bash quick_fix_build.sh

# 2. Rebuild with clean cache
cd .devcontainer
docker-compose build --no-cache datascience
```

**Permanent Fix (Recommended):**
```bash
# Move project to WSL2 native filesystem
bash fix_build_performance.sh --move-to-wsl --clean-docker --test-build

# Expected results:
# - Context transfer: <1 second (was 78s)
# - Build time: 10-15 minutes (was 19min + timeout)
# - No EOF errors
```

### ⚠️ Technical Debt Addressed
- ✅ .dockerignore now comprehensive (110 lines with comments)
- ✅ HOME variable handling defensive
- ✅ Diagnostic tooling for future debugging
- ✅ Documented Windows mount performance implications

### 🐛 Post-Build Fixes (2025-11-18 afternoon)

#### Issue 1: EOF Error During Image Export (Cosmetic)
**Error**: `failed to receive status: rpc error: code = Unavailable desc = error reading from server: EOF`
- **Cause**: Image export (15GB layers) took 85.9s on Windows mount
- **BuildKit timeout**: Status reporting connection closed after prolonged I/O
- **Impact**: None - image created successfully, containers running
- **Fix**: Cosmetic error, no action needed. Permanent fix: move to WSL filesystem

#### Issue 2: Dockerfile FROM Casing Warning
**Warning**: `FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 5)`
- **Fixed**: Changed `FROM ... as runtime` → `FROM ... AS runtime` ([Dockerfile:5](.devcontainer/Dockerfile#L5))

#### Issue 3: Healthcheck Failure - JAX GPU Detection Bug
**Error**: Container marked unhealthy despite GPU working
- **Cause**: Healthcheck checked for `"gpu"` in device string, but JAX returns `"cuda:0"`
- **Fixed**: Updated healthcheck to check for `"gpu" OR "cuda"` ([docker-compose.yml:224](.devcontainer/docker-compose.yml#L224))
- **Verification**: Manual test confirms all libraries functional (PyTorch CUDA=True, JAX GPUs=1, OpenCV=4.12.0, YOLO=OK)

### 📊 Final Build Results

**Build Performance:**
- Context transfer: 1.2s (was 78s) - **65x improvement**
- Total build time: 19.7 minutes (clean build, no cache)
- All 30/30 layers: ✅ Success
- Image export: 85.9s (slow due to Windows mount)
- Image size: ~15GB

**Container Status:**
- ✅ Datascience container: Running, healthy (after fix)
- ✅ MLflow container: Running, healthy
- ✅ GPU access: Functional (PyTorch + JAX)
- ✅ Jupyter Lab: Running on port 8888
- ✅ Computer Vision: YOLO, OpenCV, Roboflow working

**What Works:**
- All Python packages installed
- GPU passthrough functional
- CUDA 12.4, PyTorch, JAX operational
- Computer Vision stack (YOLO, OpenCV) working
- Jupyter Lab accessible
- Claude Code extensions loaded

**Known Remaining Issues:**
1. ~~Image export slow (85s) - cosmetic EOF error~~ → **ESCALATED TO CRITICAL**
2. Build still on Windows mount - ~~recommend~~ **MANDATORY** to move to WSL

### 🚨 CRITICAL UPDATE: Tar Corruption (Build 3)

**Error**: `failed to extract layer sha256:947bbd...: archive/tar: invalid tar header`

**What Changed:**
- Build 2: Export took 85.9s, cosmetic EOF, but **image created successfully** ✅
- Build 3: Export took 241.5s (2.8x worse), **tar headers corrupted**, image **NOT created** ❌

**Why Tar Corruption Occurred:**
1. Image export writes 15GB of layers to Docker storage
2. On Windows mount, this takes **241 seconds** (expected: 5-10s on WSL)
3. Prolonged I/O operations on NTFS via WSL2 9P protocol
4. **Tar format requires 512-byte header alignment**
5. Windows → WSL2 translation **corrupts byte alignment** during slow writes
6. BuildKit writes corrupted tar headers
7. Docker daemon cannot extract layers
8. Build fails completely

**Impact:**
- ❌ No image created
- ❌ Containers cannot start
- ❌ **Complete build failure**
- ❌ **No workarounds available**

**Why Workarounds Won't Work:**
- Cannot fix .dockerignore more (already at 110 lines, 1MB context)
- Cannot reduce image size (all packages are needed)
- Cannot use different storage driver (Windows mount is the problem)
- Cannot disable BuildKit (same corruption with legacy builder)

**MANDATORY Fix:**
```bash
# Move project to WSL filesystem (30 minutes)
bash fix_build_performance.sh --move-to-wsl --clean-docker --test-build

# Expected results on WSL:
# - Export time: 5-10s (was 241s)
# - No tar corruption
# - Build succeeds
```

### ✅ Fixes Implemented (Build 4 - 2025-11-18)

#### 1. Fixed HOME Variable Warning (docker-compose.yml:159)
```yaml
# Before (line 159):
- ${HOME}/.Xauthority:/root/.Xauthority:rw

# After:
- ${HOME:-/tmp}/.Xauthority:/root/.Xauthority:rw
```
- Added `/tmp` fallback for when HOME is undefined in Windows context

#### 2. Created Diagnostic Scripts
- **`diagnose_unpigz_corruption.sh`**: WSL diagnostic for storage and filesystem issues
- **`nuclear_docker_reset.ps1`**: PowerShell script for complete Docker state reset

#### 3. Resolution Steps

**Option A: Nuclear Docker Reset (Try First)**
```powershell
# From PowerShell as Admin
.\nuclear_docker_reset.ps1
# Then restart Docker Desktop
# Then rebuild from .devcontainer
```

**Option B: Move to WSL Filesystem (Permanent Fix)**
```bash
# From WSL terminal
mkdir -p ~/projects
cp -r /mnt/c/Users/ghadf/vscode_projects/docker_projects/bball_homography_pipeline ~/projects/
cd ~/projects/bball_homography_pipeline/.devcontainer
cp .env.template .env
docker-compose build --no-cache
docker-compose up -d
```

**Why Option B is Permanent**:
- Native ext4 filesystem: No 9P protocol translation
- Direct I/O: No Windows mount overhead
- Expected export time: 5-10s (vs 277.5s)
- No corruption risk from prolonged writes

**Reference:** See [CRITICAL_FIX_REQUIRED.md](CRITICAL_FIX_REQUIRED.md) for detailed migration guide

---

## [Previous] Claude Code Integration - 2025-11-18

### 🎯 Objective
Integrate Claude Code extension and configure devcontainer for AI-assisted development while preserving GPU/CV functionality

### ✅ Completed Changes

#### devcontainer.json Updates
- **Extensions Added**: `anthropic.claude-code` (primary), `eamodio.gitlens`, `esbenp.prettier-vscode`, `dbaeumer.vscode-eslint`
- **Network Capabilities**: Added `--cap-add=NET_ADMIN` and `--cap-add=NET_RAW` for Claude Code sandbox features
- **Environment Variables**: Added `CLAUDE_CONFIG_DIR=/root/.claude`, `NODE_OPTIONS=--max-old-space-size=4096`
- **Volume Mounts**: Added persistent storage for Claude config and bash history (2 new volumes)
- **Editor Settings**: Configured format-on-save for JS/TS/JSON/Markdown with Prettier, ESLint auto-fix

#### docker-compose.yml Updates
- **Capabilities**: Added `cap_add` section with NET_ADMIN and NET_RAW for Claude Code
- **Environment**: Added Claude Code config directory and Node.js memory allocation
- **Volumes**: Added `claude-bashhistory` and `claude-config` named volumes for persistence
- **Validation**: Confirmed YAML syntax valid and compatible with existing GPU configuration

#### Configuration Preservation
- ✅ All Python/Data Science extensions retained
- ✅ GPU configuration (CUDA, PyTorch, JAX) unchanged
- ✅ Computer Vision setup (YOLO, OpenCV, Roboflow) preserved
- ✅ Root user maintained (required for GPU access)
- ✅ All existing environment variables intact
- ✅ MLflow service configuration unchanged

### 🔧 Technical Details

**New Volume Mounts**:
- `claude-code-bashhistory-${devcontainerId}` → `/commandhistory` - Persistent command history across rebuilds
- `claude-code-config-${devcontainerId}` → `/root/.claude` - Claude Code settings and configuration

**Network Capabilities** (for Claude Code sandbox):
- `NET_ADMIN` - Network administration capabilities
- `NET_RAW` - Raw socket access for network operations

**Editor Enhancements**:
- Auto-format Python (Black), JavaScript/TypeScript (Prettier), JSON, Markdown
- ESLint auto-fix on save
- GitLens for advanced Git integration

### 📋 Integration Strategy
Followed non-breaking additive approach: Added Claude Code features while preserving all existing GPU/CV/ML functionality

### ⚠️ No Breaking Changes
- Existing workflows, scripts, and configurations remain functional
- GPU access and performance unaffected
- Python environment and package management unchanged

---

## [Previous] Docker Build Optimization - 2025-10-08

### 🎯 Objective
Fix `archive/tar: invalid tar header` error and reduce build time from 31min to 5-10min

### 📊 Root Cause Analysis
- **Primary**: Building from Windows mount (`/mnt/c`) causing 10-50x slower I/O
- **Secondary**: Corrupted BuildKit cache from previous failed builds
- **Contributing**: Large build context (1MB taking 470s to transfer)
- **Impact**: Tar corruption during slow layer export phase (325.2s)

### ✅ Completed
- Analyzed existing Dockerfile, docker-compose.yml, and .dockerignore configuration
- Identified WSL2 filesystem performance as critical bottleneck
- Created comprehensive root cause analysis documentation
- Created diagnostic script (`fix_docker_build.py`) with automated analysis and fixes
- Created enhanced Dockerfile (`.devcontainer/Dockerfile.optimized`) with separate cache IDs and validation
- Updated .dockerignore with comprehensive patterns (153 lines)
- Created automated build script (`build_optimized.sh`) with testing
- Generated complete implementation guide (`IMPLEMENTATION_GUIDE.md`)
- Created technical summary (`IMPLEMENTATION_SUMMARY.md`) with all changes documented
- Created quick reference guide (`DOCKER_OPTIMIZATION_README.md`)

### 📋 Ready for Implementation
All optimization files created and ready. See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for step-by-step instructions.

### 🔧 Technical Details
**Cache Strategy**: Separate cache IDs per package group to prevent corruption
- `apt-cache-${CUDA_TAG}-v2` - System packages
- `uv-pytorch-${CUDA_TAG}-v2` - PyTorch (separate from JAX)
- `uv-jax-${CUDA_TAG}-v2` - JAX and CUDA dependencies
- `uv-cv-${CUDA_TAG}-v2` - Computer Vision packages

**Build Optimizations**:
- Remove `--sbom`, `--provenance`, `--attest` to reduce export overhead
- Use `--output=type=docker` for direct Docker daemon export
- Implement timestamp logging for each build step
- Add validation checks after each package installation

---

## [Previous] Computer Vision Integration - 2025-10-07

### ✅ Completed
- Integrated YOLO v8 (ultralytics==8.3.158) for object detection
- Added OpenCV with contrib modules (opencv-contrib-python-headless>=4.10.0)
- Configured Roboflow integration (roboflow==1.2.9)
- Implemented video processing stack (moviepy, yt-dlp, ffmpeg-python)
- Added image augmentation libraries (albumentations, imgaug)
- Created GPU validation script (.devcontainer/validate_gpu.py)
- Added CV-specific test suite (.devcontainer/tests/test_yolo.py)

### 🔧 Configuration Added
- YOLO model environment variables and paths
- Roboflow API integration (workspace, project, version)
- Basketball-specific model configuration (court, player, ball detection)
- Tracking parameters (confidence thresholds, IOU, max age)
- Video I/O directory configuration

### 📦 Dependencies Added
```toml
ultralytics==8.3.158
opencv-contrib-python-headless>=4.10.0
roboflow==1.2.9
supervision>=0.26,<0.27
inference-gpu>=0.55,<0.57
moviepy==2.2.1
yt-dlp==2025.9.5
ffmpeg-python==0.2.0
albumentations>=1.3.0
imgaug>=0.4.0
```

---

## [Previous] JAX + PyTorch GPU Integration - 2025-10-06

### ✅ Completed
- Configured dual GPU framework support (PyTorch + JAX)
- Installed PyTorch with CUDA 12.4 support
- Installed JAX with CUDA 12 local support
- Upgraded CuDNN to 9.8.0.69 for JAX compatibility
- Added NVJITLINK for CUDA 12 JIT linking
- Configured memory management for RTX 4090 (24GB VRAM)

### 🔧 GPU Memory Configuration
- XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 (35% pre-allocation)
- JAX_PREALLOCATION_SIZE_LIMIT_BYTES=10737418240 (10GB limit)
- PYTORCH_CUDA_ALLOC_CONF with expandable segments
- jemalloc memory allocator for host memory efficiency

### ⚠️ Known Issues
- PyTorch and JAX cannot use GPU simultaneously without careful memory management
- Build time increased to ~30min due to large package downloads
- Cache corruption observed with long builds from Windows mounts

---

## [Previous] UV Package Manager Migration - 2025-10-05

### ✅ Completed
- Migrated from pip to UV package manager (v0.7.12)
- Created virtual environment at /app/.venv
- Configured UV project environment in docker-compose.yml
- Added UV cache persistence with named volumes

### 🎯 Benefits
- Faster dependency resolution (10-100x speedup)
- Better lockfile management (uv.lock)
- More reliable builds with cache mounts

### 🔧 Configuration
- UV_PROJECT_ENVIRONMENT=/app/.venv
- Cache mount: `/root/.cache/uv` with sharing=locked
- Shell activation helper: /app/activate_uv.sh

---

## [Previous] MLflow Integration - 2025-10-04

### ✅ Completed
- Added MLflow service to docker-compose.yml
- Configured SQLite backend store
- Set up artifact storage with volume persistence
- Fixed volume path references (../mlruns, ../mlflow_db)

### 🔧 Configuration
- MLflow UI: http://localhost:5005
- Backend: SQLite at /mlflow_db/mlflow.db
- Artifacts: /mlflow_artifacts (mapped to ../mlruns)

---

## Technical Stack Summary

### Core Environment
- **Base**: nvidia/cuda:12.4.0-devel-ubuntu22.04
- **Python**: 3.10
- **Package Manager**: UV 0.7.12

### GPU Frameworks
- **PyTorch**: 2.x with CUDA 12.4
- **JAX**: 0.4.26+ with CUDA 12 local
- **CuDNN**: 9.8.0.69
- **NVJITLINK**: 12.4+

### Computer Vision
- **YOLO**: ultralytics 8.3.158
- **OpenCV**: 4.10.0+ (headless with contrib)
- **Roboflow**: 1.2.9
- **Supervision**: 0.26
- **Inference**: GPU-enabled 0.55-0.57

### Bayesian Modeling
- **PyMC**: 5.20.0+
- **PyTensor**: 2.25.0+
- **NumPyro**: 0.18.0
- **ArviZ**: 0.14.0+
- **nutpie**: 0.7.1+

### ML/Experiment Tracking
- **MLflow**: 3.1.1+
- **Optuna**: 4.3.0+
- **XGBoost, LightGBM, CatBoost**: Latest stable

---

## Build Performance Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| First build | ~31min | 10-15min | 🔴 Needs optimization |
| Rebuild (cached) | ~15min | 2-5min | 🔴 Needs optimization |
| Context transfer | 470s (1MB) | <5s | 🔴 Critical - Windows mount |
| Layer export | 325s | <60s | 🔴 Tar corruption risk |
| Total size | ~15GB | <12GB | 🟡 Acceptable |

---

## Next Steps

### High Priority
1. Move project from `/mnt/c` to WSL filesystem (`~/projects/`)
2. Clean BuildKit cache and recreate builder
3. Update .dockerignore to reduce context size
4. Test build from WSL filesystem

### Medium Priority
1. Implement separate cache IDs for package groups
2. Add timestamp logging throughout Dockerfile
3. Create automated diagnostic script
4. Document build optimization process

### Low Priority
1. Optimize layer ordering for better caching
2. Consider multi-stage build for smaller final image
3. Explore BuildKit cache export/import for CI/CD

---

## Notes

- All timestamps in this changelog are in ISO 8601 format
- Build times measured on RTX 4090 system with 32GB RAM
- WSL2 version: Ubuntu 22.04 on Windows 11
- Docker Desktop version: Latest stable with BuildKit enabled

