# api/src/cv/diagnostics/homography_diagnostics.py
"""
COMPREHENSIVE HOMOGRAPHY DIAGNOSTICS
Systematic debugging tool following the user's requested approach:
1. Don't fill in missing values - expose the real problems
2. Dissect the problem with detailed debugging 
3. Examine output vs expected behavior
4. Review error messages in detail
5. Trace code execution step by step
6. Debug assumptions and inspect variables
7. Provide potential fixes
8. Recommend best practices
"""

from __future__ import annotations
import json
import numpy as np
import cv2
import supervision as sv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt


@dataclass
class HomographyDiagnosticResult:
    """Comprehensive diagnostic results"""
    frame_index: int
    
    # Raw Detection Data
    keypoints_raw: Dict[str, Any]
    court_vertices_raw: List[Tuple[float, float]]
    
    # Point Correspondence Analysis
    correspondence_analysis: Dict[str, Any]
    
    # Geometric Analysis  
    geometric_analysis: Dict[str, Any]
    
    # Homography Computation Analysis
    homography_analysis: Dict[str, Any]
    
    # Root Cause Assessment
    root_cause: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]


class HomographyDiagnostician:
    """
    Systematic homography debugging following the user's methodology
    """
    
    def __init__(self, court_config):
        self.court_config = court_config
        self.court_vertices = np.array(court_config.vertices, dtype=np.float32)
        
        # Expected keypoint class mappings (based on court detection model)
        self.expected_keypoint_classes = {
            0: "left_baseline_corner", 1: "left_free_throw_line", 2: "left_key",
            3: "left_opposite_key", 4: "left_opposite_free_throw", 5: "left_opposite_corner",
            6: "left_free_throw_circle_1", 7: "left_free_throw_circle_2", 8: "left_free_throw_circle_3",
            9: "center_line_left", 10: "center_circle", 11: "center_line_right",
            12: "right_free_throw_start", 13: "right_free_throw_circle", 14: "right_free_throw_end",
            15: "right_baseline_start", 16: "right_center", 17: "right_baseline_end",
            18: "right_key_start", 19: "right_basket", 20: "right_key_end",
            21: "right_free_throw_line_start", 22: "right_free_throw_line_end",
            23: "right_three_point_center", 24: "right_baseline_corner",
            25: "right_baseline_free_throw", 26: "right_baseline_key",
            27: "right_baseline_opposite_key", 28: "right_baseline_opposite_free_throw",
            29: "right_baseline_opposite_corner"
        }
    
    def diagnose_frame(
        self, 
        frame: np.ndarray, 
        court_model, 
        frame_index: int = 0,
        confidence_threshold: float = 0.5
    ) -> HomographyDiagnosticResult:
        """
        STEP 1: Comprehensive frame-by-frame diagnosis
        """
        print(f"\n{'='*80}")
        print(f"🔍 COMPREHENSIVE HOMOGRAPHY DIAGNOSIS - FRAME {frame_index}")
        print(f"{'='*80}")
        
        # STEP 1.1: Raw keypoint detection analysis
        print(f"\n🔍 STEP 1: RAW KEYPOINT DETECTION ANALYSIS")
        print(f"{'='*50}")
        
        try:
            result = court_model.infer(frame, confidence=confidence_threshold)[0]
            keypoints = sv.KeyPoints.from_inference(result)
            
            raw_analysis = self._analyze_raw_keypoints(keypoints, confidence_threshold)
            print(f"✅ Keypoint detection successful")
            
        except Exception as e:
            print(f"❌ CRITICAL: Keypoint detection failed: {e}")
            return self._create_failure_result(frame_index, "keypoint_detection_failed", str(e))
        
        # STEP 1.2: Point correspondence analysis  
        print(f"\n🔍 STEP 2: POINT CORRESPONDENCE ANALYSIS")
        print(f"{'='*50}")
        
        correspondence_analysis = self._analyze_point_correspondences(keypoints, confidence_threshold)
        
        # STEP 1.3: Geometric feasibility analysis
        print(f"\n🔍 STEP 3: GEOMETRIC FEASIBILITY ANALYSIS")
        print(f"{'='*50}")
        
        geometric_analysis = self._analyze_geometric_feasibility(
            correspondence_analysis.get("detected_points", np.array([])),
            correspondence_analysis.get("court_points", np.array([]))
        )
        
        # STEP 1.4: Homography computation analysis
        print(f"\n🔍 STEP 4: HOMOGRAPHY COMPUTATION ANALYSIS")
        print(f"{'='*50}")
        
        homography_analysis = self._analyze_homography_computation(
            correspondence_analysis.get("detected_points", np.array([])),
            correspondence_analysis.get("court_points", np.array([]))
        )
        
        # STEP 1.5: Root cause assessment
        print(f"\n🔍 STEP 5: ROOT CAUSE ASSESSMENT")
        print(f"{'='*50}")
        
        root_cause = self._assess_root_cause(
            raw_analysis, correspondence_analysis, geometric_analysis, homography_analysis
        )
        
        # STEP 1.6: Generate recommendations
        print(f"\n🔍 STEP 6: RECOMMENDATIONS")
        print(f"{'='*50}")
        
        recommendations = self._generate_recommendations(root_cause)
        
        return HomographyDiagnosticResult(
            frame_index=frame_index,
            keypoints_raw=raw_analysis,
            court_vertices_raw=[(float(v[0]), float(v[1])) for v in self.court_vertices],
            correspondence_analysis=correspondence_analysis,
            geometric_analysis=geometric_analysis,
            homography_analysis=homography_analysis,
            root_cause=root_cause,
            recommendations=recommendations
        )
    
    def _analyze_raw_keypoints(self, keypoints: sv.KeyPoints, conf_threshold: float) -> Dict[str, Any]:
        """STEP 1.1: Analyze raw keypoint detection data"""
        
        analysis = {
            "total_keypoints_detected": 0,
            "keypoints_above_threshold": 0,
            "confidence_distribution": {},
            "spatial_distribution": {},
            "detection_quality": "UNKNOWN",
            "issues_found": []
        }
        
        if keypoints.xy is None:
            analysis["issues_found"].append("CRITICAL: keypoints.xy is None")
            print("❌ CRITICAL ISSUE: keypoints.xy is None")
            return analysis
        
        if keypoints.confidence is None:
            analysis["issues_found"].append("CRITICAL: keypoints.confidence is None")
            print("❌ CRITICAL ISSUE: keypoints.confidence is None")
            return analysis
        
        # Extract data for first detection (the court)
        xy = keypoints.xy[0]  # Shape: (M, 2)
        conf = keypoints.confidence[0]  # Shape: (M,)
        
        analysis["total_keypoints_detected"] = len(xy)
        
        print(f"📊 Raw Detection Data:")
        print(f"   - Total keypoints detected: {len(xy)}")
        print(f"   - Keypoints shape: {xy.shape}")
        print(f"   - Confidence shape: {conf.shape}")
        
        # Confidence analysis
        valid_mask = conf >= conf_threshold
        analysis["keypoints_above_threshold"] = int(np.sum(valid_mask))
        
        print(f"   - Keypoints above threshold ({conf_threshold}): {analysis['keypoints_above_threshold']}")
        
        # Detailed confidence breakdown
        conf_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in conf_ranges:
            count = np.sum((conf >= low) & (conf < high))
            analysis["confidence_distribution"][f"{low}-{high}"] = int(count)
            print(f"   - Confidence {low}-{high}: {count} keypoints")
        
        # Spatial distribution analysis
        if len(xy) > 0:
            x_coords, y_coords = xy[:, 0], xy[:, 1]
            analysis["spatial_distribution"] = {
                "x_min": float(np.min(x_coords)), "x_max": float(np.max(x_coords)),
                "y_min": float(np.min(y_coords)), "y_max": float(np.max(y_coords)),
                "x_spread": float(np.max(x_coords) - np.min(x_coords)),
                "y_spread": float(np.max(y_coords) - np.min(y_coords))
            }
            
            print(f"📊 Spatial Distribution:")
            print(f"   - X range: {analysis['spatial_distribution']['x_min']:.1f} - {analysis['spatial_distribution']['x_max']:.1f} (spread: {analysis['spatial_distribution']['x_spread']:.1f})")
            print(f"   - Y range: {analysis['spatial_distribution']['y_min']:.1f} - {analysis['spatial_distribution']['y_max']:.1f} (spread: {analysis['spatial_distribution']['y_spread']:.1f})")
        
        # Quality assessment
        if analysis["keypoints_above_threshold"] >= 8:
            analysis["detection_quality"] = "GOOD"
        elif analysis["keypoints_above_threshold"] >= 4:
            analysis["detection_quality"] = "MARGINAL"
        else:
            analysis["detection_quality"] = "POOR"
            analysis["issues_found"].append(f"Insufficient high-confidence keypoints: {analysis['keypoints_above_threshold']}")
        
        print(f"🎯 Detection Quality: {analysis['detection_quality']}")
        
        return analysis
    
    def _analyze_point_correspondences(self, keypoints: sv.KeyPoints, conf_threshold: float) -> Dict[str, Any]:
        """STEP 1.2: Analyze point correspondence mapping"""
        
        analysis = {
            "correspondence_method": "SEQUENTIAL_ASSUMPTION",  # Current broken method
            "valid_correspondences": 0,
            "correspondence_quality": "UNKNOWN",
            "detected_points": np.array([]),
            "court_points": np.array([]),
            "correspondence_mapping": [],
            "issues_found": []
        }
        
        if keypoints.xy is None or keypoints.confidence is None:
            analysis["issues_found"].append("No keypoints available for correspondence")
            print("❌ No keypoints available for correspondence analysis")
            return analysis
        
        # Extract valid keypoints
        xy = keypoints.xy[0]
        conf = keypoints.confidence[0]
        valid_mask = conf >= conf_threshold
        valid_xy = xy[valid_mask]
        
        print(f"📊 Correspondence Analysis:")
        print(f"   - Valid keypoints for correspondence: {len(valid_xy)}")
        
        # CRITICAL ISSUE IDENTIFICATION: Current broken approach
        print(f"\n⚠️  ANALYZING CURRENT BROKEN APPROACH:")
        print(f"   - Current method: Take first N detected points")
        print(f"   - Map to: First N court vertices")
        print(f"   - PROBLEM: This assumes detection order = court vertex order")
        
        # Show what current broken code is doing
        if len(valid_xy) > 0:
            num_to_use = min(len(valid_xy), len(self.court_vertices))
            
            analysis["detected_points"] = valid_xy[:num_to_use]
            analysis["court_points"] = self.court_vertices[:num_to_use]
            analysis["valid_correspondences"] = num_to_use
            
            print(f"\n📋 Current Broken Correspondences (First {num_to_use}):")
            for i in range(min(5, num_to_use)):  # Show first 5
                det_pt = valid_xy[i]
                court_pt = self.court_vertices[i]
                analysis["correspondence_mapping"].append({
                    "index": i,
                    "detected": [float(det_pt[0]), float(det_pt[1])],
                    "court": [float(court_pt[0]), float(court_pt[1])],
                    "court_vertex_name": f"vertex_{i}"
                })
                print(f"   {i}: img({det_pt[0]:.1f}, {det_pt[1]:.1f}) -> court({court_pt[0]:.1f}, {court_pt[1]:.1f})")
            
            # PROBLEM IDENTIFICATION
            analysis["issues_found"].append("MAJOR: Using sequential correspondence without semantic matching")
            analysis["issues_found"].append("MAJOR: No verification that detected points correspond to assumed vertices")
            analysis["issues_found"].append("MAJOR: Detection model may output keypoints in different order than court vertices")
            
            print(f"\n❌ IDENTIFIED CORRESPONDENCE PROBLEMS:")
            for issue in analysis["issues_found"]:
                print(f"   - {issue}")
        
        # Quality assessment
        if analysis["valid_correspondences"] >= 8:
            analysis["correspondence_quality"] = "SUFFICIENT_COUNT_BUT_WRONG_MAPPING"
        elif analysis["valid_correspondences"] >= 4:
            analysis["correspondence_quality"] = "MARGINAL_COUNT_AND_WRONG_MAPPING"
        else:
            analysis["correspondence_quality"] = "INSUFFICIENT_COUNT"
        
        print(f"🎯 Correspondence Quality: {analysis['correspondence_quality']}")
        
        return analysis
    
    def _analyze_geometric_feasibility(self, detected_points: np.ndarray, court_points: np.ndarray) -> Dict[str, Any]:
        """STEP 1.3: Analyze geometric feasibility of point correspondences"""
        
        analysis = {
            "point_count": len(detected_points),
            "geometric_feasibility": "UNKNOWN",
            "spread_analysis": {},
            "collinearity_analysis": {},
            "scale_analysis": {},
            "issues_found": []
        }
        
        if len(detected_points) < 4:
            analysis["issues_found"].append(f"Insufficient points for homography: {len(detected_points)}")
            analysis["geometric_feasibility"] = "IMPOSSIBLE"
            print(f"❌ Insufficient points for homography: {len(detected_points)} < 4")
            return analysis
        
        print(f"📊 Geometric Feasibility Analysis:")
        
        # Spread analysis
        if len(detected_points) > 0:
            x_spread = float(np.max(detected_points[:, 0]) - np.min(detected_points[:, 0]))
            y_spread = float(np.max(detected_points[:, 1]) - np.min(detected_points[:, 1]))
            
            analysis["spread_analysis"] = {
                "x_spread_px": x_spread,
                "y_spread_px": y_spread,
                "total_spread": float(np.sqrt(x_spread**2 + y_spread**2))
            }
            
            print(f"   - Image spread: X={x_spread:.1f}px, Y={y_spread:.1f}px")
            
            if x_spread < 100 or y_spread < 100:
                analysis["issues_found"].append(f"Poor spread: X={x_spread:.1f}, Y={y_spread:.1f}")
                print(f"   ⚠️  Poor spread detected")
        
        # Collinearity analysis
        if len(detected_points) >= 3:
            collinearity_scores = []
            for i in range(len(detected_points) - 2):
                p1, p2, p3 = detected_points[i], detected_points[i+1], detected_points[i+2]
                
                # Calculate area of triangle (0 = collinear)
                area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2
                collinearity_scores.append(area)
            
            min_area = float(np.min(collinearity_scores))
            analysis["collinearity_analysis"] = {
                "min_triangle_area": min_area,
                "is_collinear": min_area < 10.0  # pixels
            }
            
            print(f"   - Collinearity check: min_area={min_area:.1f}px")
            
            if analysis["collinearity_analysis"]["is_collinear"]:
                analysis["issues_found"].append("Points are nearly collinear")
                print(f"   ⚠️  Points are nearly collinear")
        
        # Scale analysis between image and court spaces
        if len(detected_points) >= 2 and len(court_points) >= 2:
            img_distances = []
            court_distances = []
            
            for i in range(len(detected_points) - 1):
                img_dist = np.linalg.norm(detected_points[i+1] - detected_points[i])
                court_dist = np.linalg.norm(court_points[i+1] - court_points[i])
                img_distances.append(img_dist)
                court_distances.append(court_dist)
            
            scale_ratios = [img_d / court_d if court_d > 0 else float('inf') for img_d, court_d in zip(img_distances, court_distances)]
            
            analysis["scale_analysis"] = {
                "scale_ratios": [float(r) for r in scale_ratios if not np.isinf(r)],
                "scale_consistency": float(np.std(scale_ratios)) if scale_ratios else float('inf')
            }
            
            print(f"   - Scale consistency: std={analysis['scale_analysis']['scale_consistency']:.2f}")
            
            if analysis["scale_analysis"]["scale_consistency"] > 50:
                analysis["issues_found"].append("Inconsistent scaling between correspondences")
                print(f"   ⚠️  Inconsistent scaling detected")
        
        # Overall assessment
        if len(analysis["issues_found"]) == 0:
            analysis["geometric_feasibility"] = "FEASIBLE"
        elif len(analysis["issues_found"]) <= 2:
            analysis["geometric_feasibility"] = "MARGINAL"
        else:
            analysis["geometric_feasibility"] = "POOR"
        
        print(f"🎯 Geometric Feasibility: {analysis['geometric_feasibility']}")
        
        return analysis
    
    def _analyze_homography_computation(self, detected_points: np.ndarray, court_points: np.ndarray) -> Dict[str, Any]:
        """STEP 1.4: Analyze homography computation with detailed error tracking"""
        
        analysis = {
            "computation_attempted": False,
            "computation_successful": False,
            "computation_method": "UNKNOWN",
            "error_details": {},
            "matrix_analysis": {},
            "issues_found": []
        }
        
        if len(detected_points) < 4:
            analysis["issues_found"].append("Insufficient points for homography computation")
            print(f"❌ Cannot attempt homography: {len(detected_points)} < 4 points")
            return analysis
        
        print(f"📊 Homography Computation Analysis:")
        analysis["computation_attempted"] = True
        
        # Method 1: Direct computation (exact solution for 4 points, least squares for more)
        print(f"   🔄 Attempting direct homography computation...")
        try:
            H_direct = cv2.getPerspectiveTransform(
                detected_points[:4].astype(np.float32),
                court_points[:4].astype(np.float32)
            )
            
            if H_direct is not None:
                print(f"   ✅ Direct computation successful")
                analysis["computation_successful"] = True
                analysis["computation_method"] = "DIRECT"
                
                # Analyze matrix properties
                analysis["matrix_analysis"] = self._analyze_homography_matrix(H_direct)
                
            else:
                print(f"   ❌ Direct computation returned None")
                analysis["issues_found"].append("Direct computation returned None")
                
        except Exception as e:
            print(f"   ❌ Direct computation failed: {e}")
            analysis["error_details"]["direct_error"] = str(e)
            analysis["issues_found"].append(f"Direct computation exception: {e}")
        
        # Method 2: RANSAC (if we have more than 4 points)
        if len(detected_points) > 4:
            print(f"   🔄 Attempting RANSAC homography computation...")
            try:
                H_ransac, mask = cv2.findHomography(
                    detected_points.astype(np.float32),
                    court_points.astype(np.float32),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0
                )
                
                if H_ransac is not None:
                    inliers = np.sum(mask) if mask is not None else 0
                    print(f"   ✅ RANSAC computation successful ({inliers} inliers)")
                    
                    if not analysis["computation_successful"]:  # Only update if direct failed
                        analysis["computation_successful"] = True
                        analysis["computation_method"] = "RANSAC"
                        analysis["matrix_analysis"] = self._analyze_homography_matrix(H_ransac)
                        analysis["matrix_analysis"]["ransac_inliers"] = int(inliers)
                else:
                    print(f"   ❌ RANSAC computation returned None")
                    analysis["issues_found"].append("RANSAC computation returned None")
                    
            except Exception as e:
                print(f"   ❌ RANSAC computation failed: {e}")
                analysis["error_details"]["ransac_error"] = str(e)
                analysis["issues_found"].append(f"RANSAC computation exception: {e}")
        
        # Method 3: Check if it's a data problem vs computation problem
        if not analysis["computation_successful"]:
            print(f"   🔄 Analyzing why computation failed...")
            
            # Check for degenerate cases
            det_homogeneous = np.column_stack([detected_points, np.ones(len(detected_points))])
            court_homogeneous = np.column_stack([court_points, np.ones(len(court_points))])
            
            # Check rank of point matrices
            det_rank = np.linalg.matrix_rank(det_homogeneous)
            court_rank = np.linalg.matrix_rank(court_homogeneous)
            
            analysis["error_details"]["detected_points_rank"] = int(det_rank)
            analysis["error_details"]["court_points_rank"] = int(court_rank)
            
            print(f"   📊 Matrix ranks: detected={det_rank}, court={court_rank}")
            
            if det_rank < 3 or court_rank < 3:
                analysis["issues_found"].append(f"Degenerate point configuration: ranks {det_rank}, {court_rank}")
                print(f"   ❌ Degenerate point configuration detected")
        
        print(f"🎯 Computation Result: {'SUCCESS' if analysis['computation_successful'] else 'FAILED'}")
        
        return analysis
    
    def _analyze_homography_matrix(self, H: np.ndarray) -> Dict[str, Any]:
        """Analyze properties of computed homography matrix"""
        analysis = {}
        
        try:
            # Matrix properties
            analysis["determinant"] = float(np.linalg.det(H))
            analysis["condition_number"] = float(np.linalg.cond(H))
            
            # Check if matrix is well-conditioned
            analysis["is_well_conditioned"] = analysis["condition_number"] < 1e12
            analysis["is_invertible"] = abs(analysis["determinant"]) > 1e-10
            
            # SVD analysis
            U, s, Vh = np.linalg.svd(H)
            analysis["singular_values"] = [float(sv) for sv in s]
            analysis["rank"] = int(np.sum(s > 1e-10))
            
        except Exception as e:
            analysis["matrix_error"] = str(e)
        
        return analysis
    
    def _assess_root_cause(self, raw_analysis, correspondence_analysis, geometric_analysis, homography_analysis) -> Dict[str, Any]:
        """STEP 1.5: Determine the root cause of homography failure"""
        
        root_cause = {
            "primary_issue": "UNKNOWN",
            "contributing_factors": [],
            "severity": "UNKNOWN",
            "fix_complexity": "UNKNOWN"
        }
        
        print(f"📊 Root Cause Assessment:")
        
        # Check each stage for primary failure point
        if raw_analysis.get("detection_quality") == "POOR":
            root_cause["primary_issue"] = "INSUFFICIENT_KEYPOINT_DETECTION"
            root_cause["severity"] = "HIGH"
            root_cause["fix_complexity"] = "MEDIUM"
            print(f"   🎯 PRIMARY ISSUE: Insufficient keypoint detection")
            
        elif "MAJOR: Using sequential correspondence" in correspondence_analysis.get("issues_found", []):
            root_cause["primary_issue"] = "INCORRECT_POINT_CORRESPONDENCE"
            root_cause["severity"] = "CRITICAL"
            root_cause["fix_complexity"] = "HIGH"
            print(f"   🎯 PRIMARY ISSUE: Incorrect point correspondence mapping")
            
        elif geometric_analysis.get("geometric_feasibility") == "POOR":
            root_cause["primary_issue"] = "POOR_GEOMETRIC_CONFIGURATION"
            root_cause["severity"] = "HIGH"
            root_cause["fix_complexity"] = "MEDIUM"
            print(f"   🎯 PRIMARY ISSUE: Poor geometric configuration")
            
        elif not homography_analysis.get("computation_successful"):
            root_cause["primary_issue"] = "HOMOGRAPHY_COMPUTATION_FAILURE"
            root_cause["severity"] = "HIGH"
            root_cause["fix_complexity"] = "LOW"
            print(f"   🎯 PRIMARY ISSUE: Homography computation failure")
        
        # Collect contributing factors
        all_issues = []
        all_issues.extend(raw_analysis.get("issues_found", []))
        all_issues.extend(correspondence_analysis.get("issues_found", []))
        all_issues.extend(geometric_analysis.get("issues_found", []))
        all_issues.extend(homography_analysis.get("issues_found", []))
        
        root_cause["contributing_factors"] = list(set(all_issues))
        
        print(f"   📋 Contributing factors:")
        for factor in root_cause["contributing_factors"]:
            print(f"      - {factor}")
        
        return root_cause
    
    def _generate_recommendations(self, root_cause: Dict[str, Any]) -> List[str]:
        """STEP 1.6: Generate specific recommendations based on root cause"""
        
        recommendations = []
        primary_issue = root_cause.get("primary_issue")
        
        print(f"📋 Recommendations:")
        
        if primary_issue == "INCORRECT_POINT_CORRESPONDENCE":
            recommendations.extend([
                "CRITICAL: Replace sequential correspondence with semantic keypoint mapping",
                "Implement keypoint class ID to court vertex mapping",
                "Add correspondence validation using geometric constraints",
                "Use RANSAC to filter invalid correspondences",
                "Implement correspondence confidence scoring"
            ])
            
        elif primary_issue == "INSUFFICIENT_KEYPOINT_DETECTION":
            recommendations.extend([
                "Lower keypoint detection confidence threshold",
                "Add temporal smoothing for keypoint stability",
                "Implement keypoint interpolation for missing points",
                "Use multiple frames for robust keypoint detection"
            ])
            
        elif primary_issue == "POOR_GEOMETRIC_CONFIGURATION":
            recommendations.extend([
                "Filter correspondences by geometric spread requirements",
                "Implement collinearity checking",
                "Add point distribution quality scoring",
                "Use adaptive point selection for better geometry"
            ])
            
        elif primary_issue == "HOMOGRAPHY_COMPUTATION_FAILURE":
            recommendations.extend([
                "Add robust homography computation with RANSAC",
                "Implement multiple homography estimation methods",
                "Add homography quality validation",
                "Use iterative refinement for better accuracy"
            ])
        
        # Always add these general recommendations
        recommendations.extend([
            "Add comprehensive error logging with specific failure reasons",
            "Implement progressive fallback strategies",
            "Add visual debugging outputs for manual inspection",
            "Create automated quality metrics for homography validation"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return recommendations
    
    def _create_failure_result(self, frame_index: int, error_type: str, error_message: str) -> HomographyDiagnosticResult:
        """Create diagnostic result for complete failure cases"""
        return HomographyDiagnosticResult(
            frame_index=frame_index,
            keypoints_raw={"error": error_message},
            court_vertices_raw=[],
            correspondence_analysis={"error": error_type},
            geometric_analysis={"error": error_type},
            homography_analysis={"error": error_type},
            root_cause={"primary_issue": error_type, "severity": "CRITICAL"},
            recommendations=[f"Fix {error_type}: {error_message}"]
        )
    
    def save_diagnostic_report(self, result: HomographyDiagnosticResult, output_path: Path):
        """Save comprehensive diagnostic report"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        print(f"📁 Diagnostic report saved to: {output_path}")


def run_comprehensive_diagnostics():
    """Run the comprehensive diagnostics on a problematic frame"""
    print("🔬 RUNNING COMPREHENSIVE HOMOGRAPHY DIAGNOSTICS")
    print("="*80)
    
    try:
        from api.src.cv.config import CVConfig, load_models
        
        cfg = CVConfig()
        _, court_model = load_models(cfg)
        
        # Load problematic frame
        cap = cv2.VideoCapture(str(cfg.source_video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Failed to load test frame")
            return False
        
        # Run comprehensive diagnostics
        diagnostician = HomographyDiagnostician(cfg.court_config)
        
        result = diagnostician.diagnose_frame(
            frame=frame,
            court_model=court_model,
            frame_index=cfg.start_frame_index,
            confidence_threshold=0.5
        )
        
        # Save report
        output_dir = cfg.output_dir / "diagnostics"
        report_path = output_dir / f"homography_diagnosis_frame_{cfg.start_frame_index}.json"
        diagnostician.save_diagnostic_report(result, report_path)
        
        print(f"\n🎯 DIAGNOSIS COMPLETE")
        print(f"Primary Issue: {result.root_cause['primary_issue']}")
        print(f"Severity: {result.root_cause['severity']}")
        print(f"Fix Complexity: {result.root_cause['fix_complexity']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_diagnostics()
    exit(0 if success else 1)
