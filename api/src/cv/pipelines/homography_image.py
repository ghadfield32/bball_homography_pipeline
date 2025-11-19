# api/src/cv/pipelines/homography_image.py
"""
Fixed homography-from-image pipeline with complete integration.
All missing classes and functions are now included.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import supervision as sv
import cv2

from sports import ViewTransformer, MeasurementUnit
from sports.basketball import CourtConfiguration, draw_court

# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class Thresholds:
    """Basic threshold configuration for backward compatibility"""
    keypoint_conf: float = 0.50
    min_pairs: int = 4
    player_conf: float = 0.35
    iou_threshold: float = 0.70
    ransac_px: float = 5.0  # For backward compatibility

@dataclass
class EnhancedThresholds:
    """Enhanced thresholds with robust homography parameters"""
    keypoint_conf: float = 0.50
    min_pairs: int = 4
    player_conf: float = 0.35
    iou_threshold: float = 0.70
    
    # Robust homography parameters
    enable_robust_homography: bool = True
    ransac_reproj_thresh_court_ft: float = 1.0
    ransac_reproj_thresh_image_px: float = 5.0
    min_inlier_ratio: float = 0.5
    min_spread_px: float = 80.0
    
    # Quality gates
    homography_rmse_court_max: float = 1.5  # feet
    homography_rmse_image_max: float = 5.0  # pixels

@dataclass
class StagePaths:
    """Helper for managing output paths"""
    dir: Path
    stem: str

    def path(self, suffix: str) -> Path:
        return self.dir / f"{self.stem}-{suffix}"

# ---------------------------
# Keypoint Smoothing
# ---------------------------

class KeyPointsSmoother:
    """Temporal smoothing for keypoint stability across frames"""
    def __init__(self, length: int = 3):
        self.length = length
        self.buffer = deque(maxlen=length)

    def update(
        self,
        xy: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        conf_threshold: float = 0.0,
    ) -> np.ndarray:
        """Apply temporal smoothing to keypoints"""
        assert xy.ndim == 3 and xy.shape[0] == 1
        xy_f = xy.astype(np.float32, copy=True)

        if confidence is not None:
            assert confidence.shape[:2] == xy.shape[:2]
            mask = (confidence >= conf_threshold)[..., None]
            xy_f = np.where(mask, xy_f, np.nan)

        self.buffer.append(xy_f)
        stacked = np.stack(list(self.buffer), axis=0)

        if np.isnan(stacked).any():
            mean_xy = np.nanmean(stacked, axis=0)
        else:
            mean_xy = stacked.mean(axis=0)

        return mean_xy

# ---------------------------
# Core Detection Functions
# ---------------------------

def detect_court_keypoints_bgr(
    image_bgr: np.ndarray,
    court_model,
    keypoint_conf: float,
) -> sv.KeyPoints:
    """Detect court keypoints with confidence filtering"""
    result = court_model.infer(image_bgr, confidence=keypoint_conf)[0]
    kps = sv.KeyPoints.from_inference(result)

    if kps.confidence is None or kps.xy is None or kps.xy.size == 0:
        return kps

    # Keep only the first object (the court)
    xy = kps.xy
    conf = kps.confidence
    n, m, _ = xy.shape
    
    if n > 1:
        # Select object with highest mean confidence
        means = conf.mean(axis=1)
        idx = int(np.argmax(means))
        xy = xy[idx:idx+1]
        conf = conf[idx:idx+1]
        cls = None if kps.class_id is None else kps.class_id[idx:idx+1]
    else:
        cls = kps.class_id

    # Filter by confidence
    mask = conf[0] >= float(keypoint_conf)
    xy1 = xy[0, mask]
    conf1 = conf[0, mask]

    new_xy = xy1.reshape(1, -1, 2)
    new_conf = conf1.reshape(1, -1)
    
    return sv.KeyPoints(xy=new_xy, confidence=new_conf, class_id=cls, data={})

def detect_players_bgr(
    image_bgr: np.ndarray,
    player_model,
    confidence: float = 0.35,
    iou_threshold: float = 0.70,
) -> sv.Detections:
    """Detect players with standard inference"""
    result = player_model.infer(image_bgr, confidence=confidence, iou_threshold=iou_threshold)[0]
    return sv.Detections.from_inference(result)

# ---------------------------
# Team Classification
# ---------------------------

def _extract_torso_hsv(frame_bgr: np.ndarray, xyxy: np.ndarray) -> Tuple[float, float, float]:
    """Extract median HSV from torso region for team classification"""
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame_bgr.shape[:2]
    
    # Clip to image bounds
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h-1, y2))
    
    if x2 <= x1 or y2 <= y1:
        return float("nan"), float("nan"), float("nan")
        
    bw, bh = x2 - x1, y2 - y1
    
    # Torso band: 35%-75% vertically, 20%-80% horizontally
    tx1 = x1 + int(0.20 * bw)
    tx2 = x1 + int(0.80 * bw)
    ty1 = y1 + int(0.35 * bh)
    ty2 = y1 + int(0.75 * bh)
    
    # Ensure valid region
    tx1 = max(x1, min(x2-1, tx1))
    tx2 = max(x1+1, min(x2, tx2))
    ty1 = max(y1, min(y2-1, ty1))
    ty2 = max(y1+1, min(y2, ty2))
    
    if tx2 <= tx1 or ty2 <= ty1:
        return float("nan"), float("nan"), float("nan")
        
    patch = frame_bgr[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return float("nan"), float("nan"), float("nan")
        
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H = float(np.nanmedian(hsv[..., 0]))
    S = float(np.nanmedian(hsv[..., 1]))
    V = float(np.nanmedian(hsv[..., 2]))
    
    return H, S, V

def _kmeans1d_two_clusters(values: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Simple 1D k-means clustering for team separation"""
    vals = values[~np.isnan(values)]
    if vals.size < 2:
        return np.zeros_like(values, dtype=int)

    # Initialize centers at extremes
    c0, c1 = float(np.min(vals)), float(np.max(vals))
    labels = np.zeros_like(values, dtype=int)

    def safe_mean(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        return float(np.mean(arr)) if arr.size else float('nan')

    for _ in range(max_iter):
        d0 = np.abs(values - c0)
        d1 = np.abs(values - c1)
        lbl = (d1 < d0).astype(int)
        
        # Recompute centers
        m0 = safe_mean(values[lbl == 0])
        m1 = safe_mean(values[lbl == 1])
        new_c0 = c0 if np.isnan(m0) else m0
        new_c1 = c1 if np.isnan(m1) else m1
        
        if np.allclose([c0, c1], [new_c0, new_c1], atol=1e-3, equal_nan=True):
            labels = lbl
            break
        c0, c1 = new_c0, new_c1
        labels = lbl

    # Ensure stable ordering
    c0_final = safe_mean(values[labels == 0])
    c1_final = safe_mean(values[labels == 1])
    if not np.isnan(c0_final) and not np.isnan(c1_final) and c0_final > c1_final:
        labels = 1 - labels

    return labels

def group_players_into_teams(
    frame_bgr: np.ndarray,
    dets: sv.Detections,
    referee_class_ids: Tuple[int, ...] = (),
    referee_labels: Tuple[str, ...] = ("referee", "official")
) -> Dict[str, Dict[str, np.ndarray]]:
    """Group players into teams A, B, and referees based on jersey colors"""
    N = len(dets)
    if N == 0:
        return {
            "A": {"idx": np.array([], int), "xyxy": np.empty((0,4), float), "anchors": np.empty((0,2), float)},
            "B": {"idx": np.array([], int), "xyxy": np.empty((0,4), float), "anchors": np.empty((0,2), float)},
            "REF": {"idx": np.array([], int), "xyxy": np.empty((0,4), float), "anchors": np.empty((0,2), float)},
        }

    # Identify referees
    class_names = None
    if hasattr(dets, "data") and isinstance(dets.data, dict) and "class_name" in dets.data:
        class_names = np.array([str(x).lower() for x in dets.data["class_name"]])

    is_ref = np.zeros((N,), dtype=bool)
    if dets.class_id is not None and len(referee_class_ids) > 0:
        is_ref |= np.isin(dets.class_id, referee_class_ids)
    if class_names is not None and len(referee_labels) > 0:
        for token in referee_labels:
            is_ref |= np.char.find(class_names, token) >= 0

    # Extract hue for team clustering
    idx_nonref = np.where(~is_ref)[0]
    hues = np.full((N,), np.nan, dtype=float)
    
    if dets.xyxy is not None and len(idx_nonref) > 0:
        for i in idx_nonref:
            H, S, V = _extract_torso_hsv(frame_bgr, dets.xyxy[i])
            # Only use hue if saturation is sufficient (not grayscale)
            hues[i] = H if (not np.isnan(S) and S >= 20.0) else np.nan

    # Cluster non-referees into teams
    labels = np.zeros((N,), dtype=int)
    if len(idx_nonref) >= 2:
        labels[idx_nonref] = _kmeans1d_two_clusters(hues[idx_nonref])

    # Pack results
    def pack(mask: np.ndarray):
        idx = np.where(mask)[0]
        xyxy = dets.xyxy[idx] if dets.xyxy is not None else np.empty((0,4), float)
        anchors = dets[idx].get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        anchors = np.asarray(anchors, dtype=float) if anchors is not None else np.empty((0,2), float)
        return {"idx": idx.astype(int), "xyxy": np.asarray(xyxy, float), "anchors": anchors}

    A = pack((~is_ref) & (labels == 0))
    B = pack((~is_ref) & (labels == 1))
    R = pack(is_ref)

    return {"A": A, "B": B, "REF": R}

# ---------------------------
# Homography Computation
# ---------------------------

def _dual_ransac_filter(
    img_pts: np.ndarray,
    court_pts: np.ndarray,
    thresholds: EnhancedThresholds,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Dual-direction RANSAC filtering for robust homography"""
    dbg = {
        "used_dual_ransac": True,
        "k_in": int(len(img_pts)),
        "k_out": None,
        "reason": None,
    }

    k = int(img_pts.shape[0])
    if k < 4:
        dbg["reason"] = "too_few_for_ransac"
        return img_pts, court_pts, dbg

    # Check spread to avoid degenerate clusters
    spread_x = float(np.max(img_pts[:, 0]) - np.min(img_pts[:, 0]))
    spread_y = float(np.max(img_pts[:, 1]) - np.min(img_pts[:, 1]))
    
    if spread_x < thresholds.min_spread_px or spread_y < thresholds.min_spread_px:
        dbg["reason"] = f"degenerate_spread_x={spread_x:.1f}_y={spread_y:.1f}"
        return img_pts, court_pts, dbg

    # Image -> Court RANSAC
    H_ic, in_ic = cv2.findHomography(
        img_pts, court_pts, 
        method=cv2.RANSAC,
        ransacReprojThreshold=thresholds.ransac_reproj_thresh_court_ft,
        maxIters=2000, 
        confidence=0.995
    )
    
    if H_ic is None or in_ic is None:
        dbg["reason"] = "ransac_img2court_failed"
        return img_pts, court_pts, dbg
    
    in_ic = in_ic.ravel().astype(bool)

    # Court -> Image RANSAC
    H_ci, in_ci = cv2.findHomography(
        court_pts, img_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=thresholds.ransac_reproj_thresh_image_px,
        maxIters=2000,
        confidence=0.995
    )
    
    if H_ci is None or in_ci is None:
        dbg["reason"] = "ransac_court2img_failed"
        return img_pts, court_pts, dbg
    
    in_ci = in_ci.ravel().astype(bool)

    # Intersection of inliers
    inliers = in_ic & in_ci
    num_inliers = int(np.count_nonzero(inliers))
    dbg["k_out"] = num_inliers

    if num_inliers < thresholds.min_pairs or num_inliers < int(thresholds.min_inlier_ratio * k):
        dbg["reason"] = f"insufficient_intersection_inliers_{num_inliers}_of_{k}"
        return img_pts, court_pts, dbg

    return img_pts[inliers], court_pts[inliers], dbg

def compute_homography_robust(
    detected_on_image: np.ndarray,
    court_vertices_masked: np.ndarray,
    thresholds: EnhancedThresholds,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict]:
    """Compute homography with robust filtering and quality validation"""
    debug_info = {"stage": "start", "quality_passed": False}
    
    if len(detected_on_image) < thresholds.min_pairs:
        debug_info.update({
            "stage": "insufficient_points",
            "error": f"Only {len(detected_on_image)} points, need {thresholds.min_pairs}"
        })
        return None, None, 0, debug_info

    # Apply robust filtering if enabled
    if thresholds.enable_robust_homography:
        img_filtered, court_filtered, ransac_debug = _dual_ransac_filter(
            detected_on_image, court_vertices_masked, thresholds
        )
        debug_info.update({"ransac": ransac_debug})
        
        if ransac_debug.get("k_out", 0) < thresholds.min_pairs:
            debug_info.update({
                "stage": "ransac_failed",
                "reason": ransac_debug.get("reason", "unknown")
            })
            return None, None, 0, debug_info
    else:
        img_filtered, court_filtered = detected_on_image, court_vertices_masked

    # Compute final homography
    try:
        H, _ = cv2.findHomography(img_filtered, court_filtered, method=0)
        if H is None:
            debug_info.update({"stage": "homography_failed", "error": "cv2.findHomography returned None"})
            return None, None, 0, debug_info
        
        H_inv = np.linalg.inv(H)
        inliers_count = len(img_filtered)
        
        # Quality validation
        pred_court = cv2.perspectiveTransform(
            img_filtered.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)
        
        err_court = np.linalg.norm(pred_court - court_filtered, axis=1)
        rmse_court = float(np.sqrt(np.mean(err_court ** 2)))
        
        pred_img = cv2.perspectiveTransform(
            court_filtered.reshape(-1, 1, 2).astype(np.float32), H_inv
        ).reshape(-1, 2)
        
        err_img = np.linalg.norm(pred_img - img_filtered, axis=1)
        rmse_img = float(np.sqrt(np.mean(err_img ** 2)))
        
        quality_passed = (
            rmse_court <= thresholds.homography_rmse_court_max and 
            rmse_img <= thresholds.homography_rmse_image_max
        )
        
        debug_info.update({
            "stage": "success",
            "inliers_used": inliers_count,
            "rmse_court_ft": rmse_court,
            "rmse_image_px": rmse_img,
            "quality_passed": quality_passed
        })
        
        if not quality_passed:
            return None, None, 0, debug_info
        
        return H, H_inv, inliers_count, debug_info
        
    except Exception as e:
        debug_info.update({"stage": "computation_error", "error": str(e)})
        return None, None, 0, debug_info

# ---------------------------
# Visualization Functions
# ---------------------------

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def safe_imwrite(path: Path, image: np.ndarray) -> None:
    """Safely write image with directory creation"""
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")

def draw_keypoints_on_image(image_bgr: np.ndarray, kps: sv.KeyPoints) -> np.ndarray:
    """Draw keypoints on image"""
    annot = image_bgr.copy()
    if kps.xy is None or kps.xy.size == 0:
        return annot
    v_annot = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=6)
    return v_annot.annotate(scene=annot, key_points=kps)

def draw_homography_grid(
    image_bgr: np.ndarray,
    H_inv: np.ndarray,
    config: CourtConfiguration,
    step_ft: int = 5,
) -> np.ndarray:
    """Draw court grid on image using homography"""
    img = image_bgr.copy()
    
    def project_curve(court_xy: np.ndarray) -> np.ndarray:
        ones = np.ones((court_xy.shape[0], 1), dtype=np.float64)
        homo = np.hstack([court_xy.astype(np.float64), ones])
        mapped = (homo @ H_inv.T)
        mapped = mapped[:, :2] / mapped[:, 2:3]
        return mapped

    # Draw vertical lines
    for x_ft in range(0, int(config.length_ft) + 1, step_ft):
        y = np.linspace(0, config.width_ft, 50)
        curve_c = np.stack([np.full_like(y, x_ft), y], axis=1)
        curve_img = project_curve(curve_c)
        pts = curve_img.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), 
                     thickness=1, lineType=cv2.LINE_AA)

    # Draw horizontal lines
    for y_ft in range(0, int(config.width_ft) + 1, step_ft):
        x = np.linspace(0, config.length_ft, 50)
        curve_c = np.stack([x, np.full_like(x, y_ft)], axis=1)
        curve_img = project_curve(curve_c)
        pts = curve_img.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), 
                     thickness=1, lineType=cv2.LINE_AA)

    return img

def make_clean_court_image(
    config: CourtConfiguration,
    scale: int = 20,
    padding: int = 50,
    line_thickness: int = 4,
) -> np.ndarray:
    """Create clean court diagram"""
    return draw_court(config=config, scale=scale, padding=padding, line_thickness=line_thickness)

def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Project points using homography"""
    if pts_xy.size == 0:
        return pts_xy
    ones = np.ones((pts_xy.shape[0], 1), dtype=np.float64)
    homo = np.hstack([pts_xy.astype(np.float64), ones])
    mapped = (homo @ H.T)
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped.astype(np.float32)

# ---------------------------
# Main Pipeline Function
# ---------------------------

def process_image_to_homography(
    image: np.ndarray,
    court_model,
    player_model,
    config: CourtConfiguration,
    out_dir: Path,
    stem: str,
    thresholds: Union[Thresholds, EnhancedThresholds] = None,
    court_scale: int = 20,
    court_padding: int = 50,
    court_line_thickness: int = 4,
    save_visualizations: bool = True,
) -> Dict[str, object]:
    """
    Process single image to detect court, players, and compute homography.
    
    Returns dict with:
    - paths: Dict of saved image paths
    - homography: Dict with H matrices and quality metrics, or None if failed
    - counts: Dict with detection counts
    - team_counts: Dict with team-wise player counts
    - notes: List of processing notes
    """
    if thresholds is None:
        thresholds = EnhancedThresholds()
    elif isinstance(thresholds, Thresholds):
        # Convert old to new format
        enhanced = EnhancedThresholds()
        enhanced.keypoint_conf = thresholds.keypoint_conf
        enhanced.min_pairs = thresholds.min_pairs
        enhanced.player_conf = thresholds.player_conf
        enhanced.iou_threshold = thresholds.iou_threshold
        if hasattr(thresholds, 'ransac_px'):
            enhanced.ransac_reproj_thresh_image_px = thresholds.ransac_px
        thresholds = enhanced
    
    ensure_dir(out_dir)
    stages = StagePaths(dir=out_dir, stem=stem)
    notes = []
    paths = {}
    
    # 1. Detect court keypoints
    kps = detect_court_keypoints_bgr(image, court_model, thresholds.keypoint_conf)
    
    if save_visualizations:
        kp_overlay = draw_keypoints_on_image(image, kps)
        safe_imwrite(stages.path("keypoints.jpg"), kp_overlay)
        paths["keypoints"] = str(stages.path("keypoints.jpg"))
    
    # 2. Check if we have enough keypoints
    if kps.xy is None or kps.xy.size == 0 or kps.xy.shape[1] < thresholds.min_pairs:
        notes.append(f"Insufficient keypoints: {0 if kps.xy is None else kps.xy.shape[1]}")
        
        # Still detect players for completeness
        player_dets = detect_players_bgr(image, player_model, thresholds.player_conf, thresholds.iou_threshold)
        team_groups = group_players_into_teams(image, player_dets)
        
        return {
            "paths": paths,
            "homography": None,
            "counts": {
                "keypoints": 0 if kps.xy is None else kps.xy.shape[1],
                "players": len(player_dets)
            },
            "team_counts": {k: len(v["idx"]) for k, v in team_groups.items()},
            "notes": notes
        }
    
    # 3. Compute homography
    detected_pts = kps.xy[0, :].astype(np.float32)
    court_vertices = np.array(config.vertices, dtype=np.float32)
    
    # Match detected points with court vertices (simplified - assuming order)
    num_pts = min(len(detected_pts), len(court_vertices))
    detected_pts = detected_pts[:num_pts]
    court_vertices_matched = court_vertices[:num_pts]
    
    H, H_inv, inliers, debug = compute_homography_robust(
        detected_pts, court_vertices_matched, thresholds
    )
    
    if H is None:
        notes.append(f"Homography computation failed: {debug.get('reason', 'unknown')}")
        homography_info = None
    else:
        homography_info = {
            "H_image_to_court": H.tolist(),
            "H_court_to_image": H_inv.tolist(),
            "pairs_used": int(inliers),
            "rmse_court_ft": float(debug.get("rmse_court_ft", 0)),
            "rmse_image_px": float(debug.get("rmse_image_px", 0)),
            "quality_passed": bool(debug.get("quality_passed", False))
        }
        
        if save_visualizations and H_inv is not None:
            grid_overlay = draw_homography_grid(image, H_inv, config)
            safe_imwrite(stages.path("grid.jpg"), grid_overlay)
            paths["grid"] = str(stages.path("grid.jpg"))
    
    # 4. Detect players and classify teams
    player_dets = detect_players_bgr(image, player_model, thresholds.player_conf, thresholds.iou_threshold)
    team_groups = group_players_into_teams(image, player_dets)
    
    # 5. Create court visualization with player positions
    if save_visualizations and H is not None:
        court_img = make_clean_court_image(config, court_scale, court_padding, court_line_thickness)
        
        # Define team colors (BGR)
        team_colors = {
            "A": (255, 0, 0),      # Blue
            "B": (0, 165, 255),    # Orange
            "REF": (128, 128, 128) # Grey
        }
        
        # Project and draw each team
        for team_name, team_data in team_groups.items():
            if len(team_data["anchors"]) > 0:
                anchors_court = project_points(H, team_data["anchors"])
                color = team_colors.get(team_name, (255, 255, 255))
                
                for (x_ft, y_ft) in anchors_court:
                    # Convert to pixel coordinates
                    x_px = int(court_padding + x_ft * court_scale)
                    y_px = int(court_padding + (config.width_ft - y_ft) * court_scale)
                    
                    # Draw player marker
                    cv2.circle(court_img, (x_px, y_px), radius=8, color=color, thickness=-1)
                    cv2.circle(court_img, (x_px, y_px), radius=10, color=(0, 0, 0), thickness=2)
        
        safe_imwrite(stages.path("court.jpg"), court_img)
        paths["court"] = str(stages.path("court.jpg"))
    
    return {
        "paths": paths,
        "homography": homography_info,
        "counts": {
            "keypoints": kps.xy.shape[1],
            "players": len(player_dets)
        },
        "team_counts": {k: len(v["idx"]) for k, v in team_groups.items()},
        "notes": notes
    }
