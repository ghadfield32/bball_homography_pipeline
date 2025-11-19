# api/src/cv/pipelines/fixed_homography.py
"""
FIXED: Homography computation with proper keypoint-to-vertex correspondence mapping
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import supervision as sv
import cv2

from sports import ViewTransformer
from sports.basketball import CourtConfiguration, draw_court


@dataclass
class KeypointMapping:
    """Maps keypoint class IDs to court vertex indices"""
    # NBA court keypoint class ID -> court vertex index mapping
    # Based on basketball-court-detection-2 model output
    KEYPOINT_TO_VERTEX = {
        0: 0,   # Left baseline corner (0, 0)
        1: 1,   # Left baseline free throw (0, 3)
        2: 2,   # Left baseline key (0, 17)
        3: 3,   # Left baseline opposite key (0, 33)
        4: 4,   # Left baseline opposite free throw (0, 47)
        5: 5,   # Left baseline opposite corner (0, 50)
        6: 9,   # Left free throw circle (19, 17)
        7: 10,  # Left free throw circle (19, 25)
        8: 11,  # Left free throw circle (19, 33)
        9: 12,  # Center line left (27.4, 0)
        10: 13, # Center circle (29, 25)
        11: 14, # Center line right (27.4, 50)
        12: 15, # Right free throw start (47, 0)
        13: 16, # Right free throw circle (47, 25)
        14: 17, # Right free throw end (47, 50)
        15: 18, # Right baseline start (66.6, 0)
        16: 19, # Right center (65, 25)
        17: 20, # Right baseline end (66.6, 50)
        18: 21, # Right key start (75, 17)
        19: 22, # Right basket (75, 25)
        20: 23, # Right key end (75, 33)
        21: 24, # Right free throw line start (80, 3)
        22: 25, # Right free throw line end (80, 47)
        23: 26, # Right three point center (88.75, 25)
        24: 27, # Right baseline corner (94, 0)
        25: 28, # Right baseline free throw (94, 3)
        26: 29, # Right baseline key (94, 17)
        27: 30, # Right baseline opposite key (94, 33)
        28: 31, # Right baseline opposite free throw (94, 47)
        29: 32, # Right baseline opposite corner (94, 50)
    }

    @classmethod
    def get_valid_correspondences(
        cls, 
        keypoints: sv.KeyPoints, 
        court_vertices: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Get valid point correspondences between detected keypoints and court vertices
        
        Returns:
            detected_points: (N, 2) array of image coordinates
            court_points: (N, 2) array of court coordinates  
            used_classes: List of keypoint class IDs used
        """
        if keypoints.xy is None or keypoints.confidence is None:
            return np.empty((0, 2)), np.empty((0, 2)), []
        
        # Get keypoints for first detection (should be the court)
        xy = keypoints.xy[0]  # (M, 2)
        conf = keypoints.confidence[0]  # (M,)
        
        # Filter by confidence
        valid_mask = conf >= confidence_threshold
        valid_xy = xy[valid_mask]
        valid_conf = conf[valid_mask]
        
        # Get class IDs (assume they correspond to keypoint indices after filtering)
        # The keypoint detection model outputs keypoints in class ID order
        valid_class_ids = np.where(valid_mask)[0].tolist()
        
        detected_points = []
        court_points = []
        used_classes = []
        
        for i, class_id in enumerate(valid_class_ids):
            if class_id in cls.KEYPOINT_TO_VERTEX:
                vertex_idx = cls.KEYPOINT_TO_VERTEX[class_id]
                if vertex_idx < len(court_vertices):
                    detected_points.append(valid_xy[i])
                    court_points.append(court_vertices[vertex_idx])
                    used_classes.append(class_id)
        
        return (
            np.array(detected_points, dtype=np.float32) if detected_points else np.empty((0, 2)),
            np.array(court_points, dtype=np.float32) if court_points else np.empty((0, 2)),
            used_classes
        )


def debug_print_correspondences(
    detected_points: np.ndarray,
    court_points: np.ndarray, 
    used_classes: List[int],
    max_display: int = 10
) -> None:
    """Print detailed correspondence information for debugging"""
    print("🔍 [DEBUG] === HOMOGRAPHY COMPUTATION ===")
    print(f"🔍 [DEBUG] Input: {len(detected_points)} detected points, {len(court_points)} court vertices")
    print(f"🔍 [DEBUG] Detected points shape: {detected_points.shape}")
    print(f"🔍 [DEBUG] Court vertices shape: {court_points.shape}")
    print("🔍 [DEBUG] Point correspondences:")
    
    for i in range(min(len(detected_points), max_display)):
        img_pt = detected_points[i]
        court_pt = court_points[i]
        class_id = used_classes[i] if i < len(used_classes) else -1
        print(f"🔍 [DEBUG]   {i}: class_{class_id} img({img_pt[0]:.1f}, {img_pt[1]:.1f}) -> court({court_pt[0]:.1f}, {court_pt[1]:.1f})")


def compute_robust_homography(
    detected_points: np.ndarray,
    court_points: np.ndarray,
    min_points: int = 4,
    ransac_threshold: float = 5.0,
    max_iterations: int = 2000
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Compute homography with robust error handling and quality validation
    """
    debug_info = {
        "success": False,
        "error": None,
        "points_used": len(detected_points),
        "ransac_inliers": 0,
        "rmse": 0.0,
        "max_error": 0.0
    }
    
    # Validate input
    if len(detected_points) < min_points:
        debug_info["error"] = f"insufficient_points_{len(detected_points)}_need_{min_points}"
        return None, None, debug_info
    
    if len(detected_points) != len(court_points):
        debug_info["error"] = f"point_count_mismatch_{len(detected_points)}_vs_{len(court_points)}"
        return None, None, debug_info
    
    # Check for degenerate configurations
    if len(detected_points) >= 4:
        # Check if points are not collinear
        det_spread_x = np.max(detected_points[:, 0]) - np.min(detected_points[:, 0])
        det_spread_y = np.max(detected_points[:, 1]) - np.min(detected_points[:, 1])
        
        if det_spread_x < 50 or det_spread_y < 50:  # pixels
            debug_info["error"] = f"degenerate_spread_x_{det_spread_x:.1f}_y_{det_spread_y:.1f}"
            return None, None, debug_info
    
    try:
        print("🔍 [DEBUG] Attempting robust homography computation...")
        
        # Use RANSAC for robust estimation
        if len(detected_points) >= 4:
            H, mask = cv2.findHomography(
                detected_points,
                court_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=max_iterations,
                confidence=0.995
            )
            
            if H is None:
                debug_info["error"] = "ransac_failed"
                return None, None, debug_info
            
            inliers = mask.ravel().astype(bool)
            inlier_count = np.sum(inliers)
            debug_info["ransac_inliers"] = int(inlier_count)
            
            # Require minimum inlier ratio
            if inlier_count < min_points or inlier_count / len(detected_points) < 0.6:
                debug_info["error"] = f"insufficient_inliers_{inlier_count}_{len(detected_points)}"
                return None, None, debug_info
        
        else:
            # Direct computation for exactly 4 points
            H = cv2.getPerspectiveTransform(detected_points, court_points)
            if H is None:
                debug_info["error"] = "direct_computation_failed"
                return None, None, debug_info
            inlier_count = len(detected_points)
        
        # Compute inverse
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            debug_info["error"] = "singular_matrix"
            return None, None, debug_info
        
        # Validate homography quality
        projected_court = cv2.perspectiveTransform(
            detected_points.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)
        
        errors = np.linalg.norm(projected_court - court_points, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        max_error = float(np.max(errors))
        
        debug_info.update({
            "rmse": rmse,
            "max_error": max_error
        })
        
        # Quality thresholds
        if rmse > 3.0 or max_error > 10.0:  # feet
            debug_info["error"] = f"poor_quality_rmse_{rmse:.2f}_max_{max_error:.2f}"
            return None, None, debug_info
        
        print("✅ [SUCCESS] Robust homography computation succeeded")
        print(f"ℹ️ [INFO] Homography quality: RMSE={rmse:.2f}ft, max_error={max_error:.2f}ft")
        print(f"ℹ️ [INFO] Inliers: {inlier_count}/{len(detected_points)}")
        
        debug_info["success"] = True
        return H, H_inv, debug_info
        
    except Exception as e:
        debug_info["error"] = f"computation_exception_{str(e)}"
        print(f"❌ [ERROR] Exception during homography computation: {e}")
        return None, None, debug_info


def detect_and_compute_homography(
    image: np.ndarray,
    court_model,
    court_config: CourtConfiguration,
    confidence_threshold: float = 0.5,
    min_keypoints: int = 4
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Detect court keypoints and compute homography with proper correspondence
    """
    debug_info = {
        "keypoints_detected": 0,
        "keypoints_used": 0,
        "correspondences_found": 0,
        "homography_success": False
    }
    
    try:
        # Detect court keypoints
        result = court_model.infer(image, confidence=confidence_threshold)[0]
        keypoints = sv.KeyPoints.from_inference(result)
        
        if keypoints.xy is None or keypoints.confidence is None:
            debug_info["error"] = "no_keypoints_detected"
            return None, None, debug_info
        
        debug_info["keypoints_detected"] = keypoints.xy.shape[1]
        
        # Get valid correspondences using proper mapping
        court_vertices = np.array(court_config.vertices, dtype=np.float32)
        detected_points, court_points, used_classes = KeypointMapping.get_valid_correspondences(
            keypoints, court_vertices, confidence_threshold
        )
        
        debug_info.update({
            "keypoints_used": len(detected_points),
            "correspondences_found": len(used_classes)
        })
        
        if len(detected_points) < min_keypoints:
            debug_info["error"] = f"insufficient_correspondences_{len(detected_points)}"
            return None, None, debug_info
        
        # Debug print correspondences
        debug_print_correspondences(detected_points, court_points, used_classes)
        
        # Compute homography
        H, H_inv, homography_debug = compute_robust_homography(
            detected_points, court_points
        )
        
        debug_info.update(homography_debug)
        debug_info["homography_success"] = (H is not None)
        
        return H, H_inv, debug_info
        
    except Exception as e:
        debug_info["error"] = f"detection_exception_{str(e)}"
        print(f"❌ [ERROR] Exception in homography detection: {e}")
        return None, None, debug_info


def process_frame_with_fixed_homography(
    frame: np.ndarray,
    court_model,
    player_model,
    court_config: CourtConfiguration,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Process a single frame with the fixed homography pipeline
    """
    result = {
        "homography_success": False,
        "H_image_to_court": None,
        "H_court_to_image": None,
        "debug_info": {},
        "player_count": 0,
        "keypoint_count": 0
    }
    
    # Compute homography
    H, H_inv, debug_info = detect_and_compute_homography(
        frame, court_model, court_config, confidence_threshold
    )
    
    result.update({
        "homography_success": (H is not None),
        "H_image_to_court": H,
        "H_court_to_image": H_inv,
        "debug_info": debug_info,
        "keypoint_count": debug_info.get("keypoints_detected", 0)
    })
    
    # Detect players
    try:
        player_result = player_model.infer(frame, confidence=confidence_threshold)[0]
        player_detections = sv.Detections.from_inference(player_result)
        result["player_count"] = len(player_detections)
    except Exception as e:
        print(f"❌ [ERROR] Player detection failed: {e}")
        result["player_count"] = 0
    
    return result


# Test function to validate the fix
def test_fixed_homography():
    """Test the fixed homography computation"""
    print("🧪 Testing fixed homography computation...")
    
    try:
        from api.src.cv.config import CVConfig, load_models
        
        # Setup
        cfg = CVConfig()
        player_model, court_model = load_models(cfg)
        
        # Load test frame
        cap = cv2.VideoCapture(str(cfg.source_video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Failed to load test frame")
            return False
        
        print(f"✅ Loaded test frame: {frame.shape}")
        
        # Process frame
        result = process_frame_with_fixed_homography(
            frame, court_model, player_model, cfg.court_config
        )
        
        # Display results
        print(f"\n📊 Fixed Homography Test Results:")
        print(f"  - Keypoints detected: {result['keypoint_count']}")
        print(f"  - Players detected: {result['player_count']}")
        print(f"  - Homography success: {result['homography_success']}")
        
        debug = result['debug_info']
        if debug:
            print(f"  - Correspondences found: {debug.get('correspondences_found', 0)}")
            print(f"  - RANSAC inliers: {debug.get('ransac_inliers', 0)}")
            print(f"  - RMSE: {debug.get('rmse', 0):.2f} ft")
        
        if result['homography_success']:
            print("✅ Homography computation SUCCESSFUL!")
            return True
        else:
            print(f"❌ Homography failed: {debug.get('error', 'unknown')}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fixed_homography()
    exit(0 if success else 1)
