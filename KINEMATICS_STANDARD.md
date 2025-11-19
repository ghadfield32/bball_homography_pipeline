# Kinematics Standardization

**Last Updated**: 2025-11-19

A unified coordinate system and skeleton format for basketball CV + biomechanics analysis, aligned with SPL-Open-Data and mplbasketball.

---

## 1. Coordinate Systems

### 1.1 Image Space

| Property | Value |
|----------|-------|
| Symbol | `(u_px, v_px)` |
| Units | Pixels |
| Origin | Top-left of frame |
| Axes | `u` right, `v` down |

### 1.2 Court Space (2D)

| Property | Value |
|----------|-------|
| Symbol | `(x_court_m, y_court_m)` |
| Units | Metres (expose feet via conversion) |
| Origin | Court center |
| X-axis | Length direction (towards home basket) |
| Y-axis | Width direction (towards scorer's table) |

**Court Dimensions:**
- NBA: 28.65m × 15.24m (94ft × 50ft)
- NCAA: 28.65m × 15.24m
- FIBA: 28.0m × 15.0m

### 1.3 World Space (3D)

| Property | Value |
|----------|-------|
| Symbol | `(x_world_m, y_world_m, z_world_m)` |
| Units | Metres |
| Origin | Court center, floor plane |
| X-axis | Same as `x_court` |
| Y-axis | Same as `y_court` |
| Z-axis | Height above floor (up) |

---

## 2. Skeleton & Joint Names

### 2.1 Canonical Joint List

```python
CANONICAL_JOINTS = [
    "HEAD", "NECK",
    "R_SHOULDER", "L_SHOULDER",
    "R_ELBOW", "L_ELBOW",
    "R_WRIST", "L_WRIST",
    "R_HIP", "L_HIP",
    "R_KNEE", "L_KNEE",
    "R_ANKLE", "L_ANKLE",
    # Extended (optional)
    "CHEST", "PELVIS", "R_HAND", "L_HAND", "BALL"
]
```

### 2.2 COCO → Canonical Mapping

| COCO Index | COCO Name | Canonical |
|------------|-----------|-----------|
| 0 | nose | HEAD |
| 5 | left_shoulder | L_SHOULDER |
| 6 | right_shoulder | R_SHOULDER |
| 7 | left_elbow | L_ELBOW |
| 8 | right_elbow | R_ELBOW |
| 9 | left_wrist | L_WRIST |
| 10 | right_wrist | R_WRIST |
| 11 | left_hip | L_HIP |
| 12 | right_hip | R_HIP |
| 13 | left_knee | L_KNEE |
| 14 | right_knee | R_KNEE |
| 15 | left_ankle | L_ANKLE |
| 16 | right_ankle | R_ANKLE |

HEAD is computed as average of face landmarks (nose, eyes, ears).

### 2.3 SPL/Visual3D → Canonical Mapping

| SPL Name | Canonical |
|----------|-----------|
| RSHO | R_SHOULDER |
| LSHO | L_SHOULDER |
| REL | R_ELBOW |
| LEL | L_ELBOW |
| RWR | R_WRIST |
| LWR | L_WRIST |
| RHIP | R_HIP |
| LHIP | L_HIP |
| RKNE | R_KNEE |
| LKNE | L_KNEE |
| RANK | R_ANKLE |
| LANK | L_ANKLE |

---

## 3. Data Schema

### 3.1 JointCoordinate Table

Core table structure (one row per frame × player × joint):

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Video/trial identifier |
| `frame_idx` | int | Frame number |
| `timestamp_s` | float | Timestamp in seconds |
| `player_id` | int | Track ID |
| `team_id` | int | Team identifier |
| `jersey_number` | str | Jersey number (if available) |
| `joint` | str | Canonical joint name |
| `u_px` | float | Image x-coordinate |
| `v_px` | float | Image y-coordinate |
| `x_court_m` | float | Court x (metres) |
| `y_court_m` | float | Court y (metres) |
| `x_world_m` | float | World x (metres) |
| `y_world_m` | float | World y (metres) |
| `z_world_m` | float | World z (metres) |
| `joint_confidence` | float | Detection confidence (0-1) |
| `homography_quality` | float | Homography RMSE |
| `homography_segment_id` | int | Camera segment |
| `shot_id` | int | Shot event ID |
| `possession_id` | int | Possession ID |

### 3.2 Export Formats

- **Parquet** (default): Efficient columnar storage
- **CSV**: Human-readable, spreadsheet compatible

---

## 4. Height Estimation

When depth/3D data isn't available, we estimate joint heights using anthropometric ratios:

| Joint | Height Ratio |
|-------|-------------|
| HEAD | 0.93 |
| NECK | 0.87 |
| SHOULDER | 0.82 |
| ELBOW | 0.63 |
| WRIST | 0.47 |
| HIP | 0.53 |
| KNEE | 0.29 |
| ANKLE | 0.04 |

Default player height: 2.01m (~6'7" NBA average)

Formula: `z_world = ratio × player_height`

---

## 5. Usage

### 5.1 CV Pipeline → Standardized Output

```python
from api.src.cv.kinematics_standardization import (
    create_kinematics_standardizer,
    JointCoordinate,
)

# Create standardizer
standardizer = create_kinematics_standardizer(cfg)

# Convert pose keypoints
joints = standardizer.standardize_keypoints(
    video_id="game_001",
    frame_idx=100,
    timestamp_s=3.33,
    player_id=5,
    keypoints=pose_keypoints,  # (17, 2) COCO format
    H=homography_matrix,
    team_id=1,
    jersey_number="23",
)

# Export
standardizer.export_to_parquet(joints, "output/game_001_joints.parquet")
```

### 5.2 SPL Data → Standardized Format

```python
from api.src.biomech.spl_adapter import create_spl_adapter

# Create adapter
adapter = create_spl_adapter(cfg)

# Optionally set precise transform from landmarks
adapter.set_transform_from_landmarks(
    spl_landmarks=spl_ankle_positions,
    world_landmarks=court_ankle_positions,
)

# Load trial
joints = adapter.load_trial_csv("data/trial_001.csv", trial_id=1)

# Export
adapter.export_to_parquet(joints, "output/spl_trial_001.parquet")
```

### 5.3 Unified Analysis

Both CV and SPL data share the same schema:

```python
import pandas as pd

# Load both sources
cv_df = pd.read_parquet("output/game_001_joints.parquet")
spl_df = pd.read_parquet("output/spl_trial_001.parquet")

# Combine for comparison
combined = pd.concat([cv_df, spl_df])

# Analyze release point
release = combined[
    (combined["joint"] == "R_WRIST") &
    (combined["shot_id"] == 1)
]
```

---

## 6. Configuration

### 6.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_KINEMATICS_EXPORT` | 0 | Enable export |
| `KINEMATICS_COURT_TYPE` | NBA | Court type |
| `KINEMATICS_UNITS` | metres | Output units |
| `DEFAULT_PLAYER_HEIGHT` | 2.01 | Player height (m) |
| `ESTIMATE_Z_FROM_RATIOS` | 1 | Use anthropometric z |
| `SPL_FPS` | 120.0 | SPL mocap frame rate |
| `KINEMATICS_FORMAT` | parquet | Export format |

### 6.2 Config Parameters

```python
# In CVConfig
enable_kinematics_export: bool = True
kinematics_court_type: str = "NBA"
kinematics_output_units: str = "metres"
default_player_height_m: float = 2.01
estimate_z_from_ratios: bool = True
spl_fps: float = 120.0
kinematics_format: str = "parquet"
```

---

## 7. SPL Transform Calibration

### 7.1 Default Transform

Assumes SPL lab coordinates:
- X = forward (towards basket)
- Y = left
- Z = up
- Units: mm
- Origin: free-throw line

Transform to world coordinates:
- Flip Y axis
- Translate origin to court center
- Scale mm → metres

### 7.2 Calibration from Landmarks

For precise alignment, use known landmarks:

```python
# SPL coordinates of landmarks (in mm)
spl_landmarks = np.array([
    [0, 500, 0],      # Right ankle at stance
    [0, -500, 0],     # Left ankle at stance
    [4570, 0, 3050],  # Approximate hoop position
])

# Corresponding world coordinates (in metres)
world_landmarks = np.array([
    [-9.75, 0.5, 0],     # Right ankle
    [-9.75, -0.5, 0],    # Left ankle
    [-14.32, 0, 3.05],   # Hoop
])

adapter.set_transform_from_landmarks(spl_landmarks, world_landmarks)
```

---

## 8. Validation

### 8.1 Round-Trip Test

```python
# Transform SPL → world → SPL
original = np.array([1000, 500, 1800])  # mm
world = adapter.transform.transform(original)
recovered = adapter.transform.inverse_transform(world)

assert np.allclose(original * 0.001, recovered * 0.001, atol=1e-6)
```

### 8.2 Sanity Checks

- Ankle z ≈ 0 (standing)
- Shoulder height ≈ 1.6m (for 2m player)
- Points within court bounds

### 8.3 Visual Overlay

Plot court traces and compare SPL vs CV for the same shot type.

---

## 9. Change Log

| Date | Change |
|------|--------|
| 2025-11-19 | Initial standard: defined coordinate systems, canonical joints, data schema |
| 2025-11-19 | Added SPL adapter with Kabsch transform alignment |
| 2025-11-19 | Added config parameters for kinematics export |

---

## 10. Open TODOs

- [ ] Refine SPL → world rotation after inspecting Visual3D docs
- [ ] Integrate exact mplbasketball API calls for reanimation
- [ ] Add C3D loader for raw SPL files
- [ ] Implement automatic camera segment detection for homography
- [ ] Add depth estimation from multi-view or monocular depth models
