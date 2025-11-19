# CV Pipeline Log

**Purpose**: Compact tracking of profile/feature usage and tuning decisions. One-two liners per entry.

---

## [Profiles]

- 2025-11-19 — Added CV profiles: fast_debug, tracking_only, full_biomech, live_stream
- 2025-11-19 — Profile presets auto-configure: tracking, homography, pose, arc, OCR, re-ID, streaming
- 2025-11-19 — Use `apply_profile_to_env("full_biomech")` before `load_cv_config()` for programmatic switching

## [Detection Models]

- 2025-11-19 — Player: basketball-player-detection-3-ycjdo/4, Court: basketball-court-detection-2/14
- 2025-11-19 — Ball: basketball-detection/1, Court KPs: basketball-court-keypoints/1
- 2025-11-19 — Jersey: basketball-jersey-numbers/1, Rim: basketball-rim-detection/1

## [Tracking]

- 2025-11-19 — ByteTrack tuned: activation=0.20, lost_buffer=60, match=0.6, min_frames=2
- 2025-11-19 — SAM2 tracking option added (enable_sam2_tracking), default off

## [Homography & ViewTransformer]

- 2025-11-19 — Segment-level homography with RANSAC + semantic constraints
- 2025-11-19 — Constraints: line_collinearity_threshold=0.5ft, arc_radius_threshold=1.0ft
- 2025-11-19 — Three-point arc radius: 23.75ft (NBA standard)

## [Jersey OCR & Identity]

- 2025-11-19 — OCR backends: easyocr (default), paddleocr, smolvlm2, yolo_classifier
- 2025-11-19 — Jersey region: top=0.1, bottom=0.5, left=0.2, right=0.8 (relative to bbox)
- 2025-11-19 — Majority voting window: 10 frames for stable number assignment

## [Pose & Biomechanics]

- 2025-11-19 — YOLO-pose model: yolov8n-pose (options: n/s/m)
- 2025-11-19 — Shot form analysis: elbow angle, joint sequencing
- 2025-11-19 — Zone detection: 3-second violation threshold = 90 frames at 30fps

## [Pose & YOLO Integration]

- 2025-11-19 — Standardized pose contract: get_pose_dict_for_tracks() → {track_id: {joint: (u, v, conf, vis)}}
- 2025-11-19 — Canonical joint naming: JOINT_ID_TO_CANONICAL maps COCO indices → R_WRIST/L_KNEE/HEAD/etc
- 2025-11-19 — HEAD derived from NOSE keypoint; duplicate face landmarks (eyes, ears) skipped
- 2025-11-19 — Homography bridge: per-joint image coords projected via H → (x_court, y_court)
- 2025-11-19 — Kinematics export: one JointCoordinate row per joint per frame per track

## [Shot Arc Analysis]

- 2025-11-19 — Ball tracking for trajectory: min_points=5, velocity_window=3, smoothing=3
- 2025-11-19 — Metrics: release angle, entry angle, apex height, velocity
- 2025-11-19 — Physical constants: rim_height=10ft, ball_diameter=0.78ft

## [Visual Re-ID]

- 2025-11-19 — SigLIP model: google/siglip-base-patch16-224, embedding_dim=768
- 2025-11-19 — Similarity threshold: 0.85, max_age=90 frames, history=50 embeddings

## [API & Streaming]

- 2025-11-19 — FastAPI: port 8000, workers 1, max_video_size 500MB
- 2025-11-19 — WebSocket: port 8765, frame_skip=1, quality=80

## [Kinematics Standardization]

- 2025-11-19 — Coordinate systems: image (px), court (m), world (m, z up)
- 2025-11-19 — Joint naming: COCO → canonical (R_WRIST, L_KNEE, etc.)
- 2025-11-19 — SPL adapter: Kabsch transform for lab → court alignment
- 2025-11-19 — Height estimation: anthropometric ratios, default player=2.01m
- 2025-11-19 — Pipeline integration: process_video_enhanced accumulates JointCoordinate per frame
- 2025-11-19 — Export: KINEMATICS_FORMAT=parquet|csv, output to kinematics_output_dir
- 2025-11-19 — SPL helper: spl_trial_to_parquet() for direct trial → parquet conversion

## [Validation & Errors]

- 2025-11-19 — Config validation: fail-fast on missing required params
- 2025-11-19 — Checks: ball_model for arc, siglip_model for re-ID, pose_model for pose
- 2025-11-19 — Kinematics: video_fps > 0 required, format must be parquet|csv

## [Open TODOs]

- [ ] Tune ByteTrack further with real game footage (screens, paint occlusions)
- [ ] Test jersey OCR accuracy across different broadcast qualities
- [ ] Calibrate SPL transform with actual Visual3D data
- [ ] Add depth estimation for better z_world computation
- [ ] Benchmark profile performance (frames/sec per profile)
