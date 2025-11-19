# api/src/cv/pipelines/fixed_video_pipeline.py
"""
FIXED: Complete video pipeline with corrected homography computation
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import json
import numpy as np
import supervision as sv
import cv2
from tqdm import tqdm

from api.src.cv.config import CVConfig, load_models, court_base_image
from api.src.cv.shot_tracker import ShotEventTracker, Shot
from api.src.cv.pipelines.fixed_homography import (
    detect_and_compute_homography,
    KeypointMapping,
    debug_print_correspondences
)


class DetailedHomographyDebugger:
    """Enhanced debugging specifically for homography issues"""
    
    def __init__(self):
        self.frame_stats = []
        self.homography_failures = []
        self.successful_computations = []
        
    def log_frame(self, frame_idx: int, debug_info: Dict, success: bool):
        """Log detailed frame-by-frame homography information"""
        stats = {
            "frame": frame_idx,
            "success": success,
            "keypoints_detected": debug_info.get("keypoints_detected", 0),
            "correspondences_found": debug_info.get("correspondences_found", 0),
            "error": debug_info.get("error", None),
            "rmse": debug_info.get("rmse", 0.0),
            "ransac_inliers": debug_info.get("ransac_inliers", 0)
        }
        
        self.frame_stats.append(stats)
        
        if success:
            self.successful_computations.append(frame_idx)
        else:
            self.homography_failures.append({
                "frame": frame_idx,
                "error": debug_info.get("error", "unknown"),
                "details": debug_info
            })
    
    def print_summary(self):
        """Print comprehensive debugging summary"""
        total_frames = len(self.frame_stats)
        successful_frames = len(self.successful_computations)
        
        print(f"\n🔍 [HOMOGRAPHY DEBUG] === PROCESSING SUMMARY ===")
        print(f"🔍 [HOMOGRAPHY DEBUG] Total frames processed: {total_frames}")
        print(f"🔍 [HOMOGRAPHY DEBUG] Successful homographies: {successful_frames}/{total_frames} ({successful_frames/total_frames*100:.1f}%)")
        
        if self.homography_failures:
            print(f"🔍 [HOMOGRAPHY DEBUG] Failed frames: {len(self.homography_failures)}")
            
            # Analyze failure reasons
            failure_reasons = {}
            for failure in self.homography_failures:
                reason = failure["error"]
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            print(f"🔍 [HOMOGRAPHY DEBUG] Failure reasons:")
            for reason, count in failure_reasons.items():
                print(f"🔍 [HOMOGRAPHY DEBUG]   - {reason}: {count} frames")
        
        # Keypoint statistics
        if self.frame_stats:
            keypoints_detected = [s["keypoints_detected"] for s in self.frame_stats]
            correspondences_found = [s["correspondences_found"] for s in self.frame_stats]
            
            print(f"🔍 [HOMOGRAPHY DEBUG] Keypoint statistics:")
            print(f"🔍 [HOMOGRAPHY DEBUG]   - Avg keypoints detected: {np.mean(keypoints_detected):.1f}")
            print(f"🔍 [HOMOGRAPHY DEBUG]   - Avg correspondences found: {np.mean(correspondences_found):.1f}")
            print(f"🔍 [HOMOGRAPHY DEBUG]   - Min correspondences: {np.min(correspondences_found)}")
            print(f"🔍 [HOMOGRAPHY DEBUG]   - Max correspondences: {np.max(correspondences_found)}")


@dataclass 
class FixedVideoResult:
    """Result container for the fixed video pipeline"""
    video_path: str
    output_video_path: str
    court_video_path: str
    total_frames: int
    processed_frames: int
    successful_homographies: int
    shots_detected: int
    team_stats: Dict
    processing_time: float
    success: bool
    notes: List[str]
    debug_summary: Dict


class FixedVideoPipeline:
    """
    FIXED: Video pipeline with proper homography computation and debugging
    """
    
    def __init__(self, cfg: CVConfig):
        self.cfg = cfg
        self.player_model, self.court_model = load_models(cfg)
        
        # Enhanced debugging
        self.homography_debugger = DetailedHomographyDebugger()
        
        # Colors for visualization
        self.team_colors = {
            "A": (255, 0, 0),      # Blue in BGR
            "B": (0, 165, 255),    # Orange in BGR  
            "REF": (128, 128, 128) # Gray
        }
        
        print("[FixedPipeline] Initialized with enhanced homography debugging")
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        max_frames: Optional[int] = None
    ) -> FixedVideoResult:
        """Process video with fixed homography pipeline"""
        
        start_time = time.time()
        video_path = Path(video_path)
        
        if output_dir is None:
            output_dir = self.cfg.output_dir / "fixed_pipeline"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup output paths
        stem = video_path.stem
        output_video_path = output_dir / f"{stem}_fixed_tracked.mp4"
        court_video_path = output_dir / f"{stem}_fixed_court.mp4"
        
        print(f"[FixedPipeline] Processing: {video_path}")
        print(f"[FixedPipeline] Output: {output_video_path}")
        
        # Video setup
        video_info = sv.VideoInfo.from_video_path(str(video_path))
        total_frames = getattr(video_info, 'total_frames', getattr(video_info, 'frame_count', 1000))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Initialize tracking
        shot_tracker = ShotEventTracker(
            reset_time_frames=int(video_info.fps * 1.7),
            minimum_frames_between_starts=int(video_info.fps * 0.5),
            cooldown_frames_after_made=int(video_info.fps * 0.5)
        )
        
        # Court visualization setup
        court_base = court_base_image(self.cfg)
        court_h, court_w = court_base.shape[:2]
        court_video_info = sv.VideoInfo(
            width=court_w, height=court_h, fps=video_info.fps, total_frames=total_frames
        )
        
        # Storage
        all_shots = []
        successful_homographies = 0
        processed_frames = 0
        notes = []
        
        # Processing loop
        frame_generator = sv.get_video_frames_generator(str(video_path))
        
        with sv.VideoSink(str(output_video_path), video_info) as sink, \
             sv.VideoSink(str(court_video_path), court_video_info) as court_sink:
            
            for frame_idx, frame in enumerate(tqdm(
                frame_generator, total=total_frames, desc="Processing with fixed homography"
            )):
                if max_frames and frame_idx >= max_frames:
                    break
                
                try:
                    frame_result = self._process_single_frame(
                        frame, frame_idx, shot_tracker, court_base, all_shots
                    )
                    
                    if frame_result["homography_success"]:
                        successful_homographies += 1
                    
                    sink.write_frame(frame_result["annotated_frame"])
                    court_sink.write_frame(frame_result["court_frame"])
                    
                    processed_frames += 1
                    
                except Exception as e:
                    print(f"[FixedPipeline] Frame {frame_idx} error: {e}")
                    notes.append(f"Frame {frame_idx} failed: {str(e)}")
                    
                    # Write original frames to maintain continuity
                    sink.write_frame(frame)
                    court_sink.write_frame(court_base)
                    processed_frames += 1
        
        # Print debugging summary
        self.homography_debugger.print_summary()
        
        # Calculate team statistics (simplified)
        team_stats = {
            "TEAM_A": {
                "total_shots": len([s for s in all_shots if s.team == 0]),
                "made": len([s for s in all_shots if s.team == 0 and s.result]),
                "missed": len([s for s in all_shots if s.team == 0 and not s.result]),
            },
            "TEAM_B": {
                "total_shots": len([s for s in all_shots if s.team == 1]),
                "made": len([s for s in all_shots if s.team == 1 and s.result]),
                "missed": len([s for s in all_shots if s.team == 1 and not s.result]),
            }
        }
        
        processing_time = time.time() - start_time
        
        # Create result
        result = FixedVideoResult(
            video_path=str(video_path),
            output_video_path=str(output_video_path),
            court_video_path=str(court_video_path),
            total_frames=total_frames,
            processed_frames=processed_frames,
            successful_homographies=successful_homographies,
            shots_detected=len(all_shots),
            team_stats=team_stats,
            processing_time=processing_time,
            success=True,
            notes=notes,
            debug_summary={
                "homography_success_rate": successful_homographies / processed_frames if processed_frames > 0 else 0,
                "frames_per_second": processed_frames / processing_time if processing_time > 0 else 0,
                "average_keypoints": np.mean([s["keypoints_detected"] for s in self.homography_debugger.frame_stats]) if self.homography_debugger.frame_stats else 0
            }
        )
        
        # Save debug info
        debug_path = output_dir / f"{stem}_debug.json"
        with open(debug_path, 'w') as f:
            json.dump({
                "frame_stats": self.homography_debugger.frame_stats,
                "successful_frames": self.homography_debugger.successful_computations,
                "failure_analysis": [{"frame": f["frame"], "error": f["error"]} for f in self.homography_debugger.homography_failures]
            }, f, indent=2)
        
        print(f"\n[FixedPipeline] Processing complete!")
        print(f"  - Processed: {processed_frames} frames")
        print(f"  - Successful homographies: {successful_homographies}/{processed_frames} ({successful_homographies/processed_frames*100:.1f}%)")
        print(f"  - Shots detected: {len(all_shots)}")
        print(f"  - Processing time: {processing_time:.1f}s")
        print(f"  - Debug info saved to: {debug_path}")
        
        return result
    
    def _process_single_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        shot_tracker: ShotEventTracker,
        court_base: np.ndarray,
        all_shots: List[Shot]
    ) -> Dict:
        """Process single frame with fixed homography computation"""
        
        result = {
            "homography_success": False,
            "annotated_frame": frame.copy(),
            "court_frame": court_base.copy()
        }
        
        # FIXED: Use the corrected homography detection
        H, H_inv, debug_info = detect_and_compute_homography(
            frame, 
            self.court_model, 
            self.cfg.court_config,
            confidence_threshold=0.5,
            min_keypoints=4
        )
        
        # Log for debugging
        self.homography_debugger.log_frame(frame_idx, debug_info, H is not None)
        
        result["homography_success"] = (H is not None)
        
        # If homography successful, continue with player detection and projection
        if H is not None:
            try:
                # Detect players
                player_result = self.player_model.infer(frame, confidence=0.3)[0]
                player_detections = sv.Detections.from_inference(player_result)
                
                # Detect shot events
                jump_shot_mask = player_detections.class_id == 5
                layup_mask = player_detections.class_id == 6
                basket_mask = player_detections.class_id == 1
                
                has_jump_shot = np.any(jump_shot_mask)
                has_layup = np.any(layup_mask)
                has_basket = np.any(basket_mask)
                
                # Update shot tracker
                events = shot_tracker.update(
                    frame_index=frame_idx,
                    has_jump_shot=has_jump_shot,
                    has_layup_dunk=has_layup,
                    has_ball_in_basket=has_basket
                )
                
                # Process shot events
                for event in events:
                    if event.event_type in ["MADE", "MISSED"]:
                        # Find shot position
                        shot_detections = player_detections[jump_shot_mask | layup_mask]
                        if len(shot_detections) > 0:
                            anchors = shot_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                            if len(anchors) > 0:
                                # Project to court
                                shot_image = np.array([anchors[0]], dtype=np.float32)
                                shot_court = cv2.perspectiveTransform(
                                    shot_image.reshape(-1, 1, 2), H
                                ).reshape(-1, 2)
                                
                                x_court, y_court = shot_court[0]
                                
                                # Calculate distance to basket
                                basket_pos = self.cfg.court_config.vertices[10]  # Approximate basket
                                distance = np.linalg.norm([x_court - basket_pos[0], y_court - basket_pos[1]])
                                
                                # Create shot record
                                shot = Shot(
                                    x=float(x_court),
                                    y=float(y_court), 
                                    distance=float(distance),
                                    result=(event.event_type == "MADE"),
                                    team=0,  # Simplified
                                    frame_index=frame_idx,
                                    shot_type=event.shot_type or "UNKNOWN"
                                )
                                
                                all_shots.append(shot)
                                print(f"🏀 [SHOT] Frame {frame_idx}: {event.event_type} at ({x_court:.1f}, {y_court:.1f})")
                
                # Create court visualization with player positions
                court_visual = court_base.copy()
                
                # Project all players to court
                if len(player_detections) > 0:
                    player_anchors = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    if len(player_anchors) > 0:
                        players_court = cv2.perspectiveTransform(
                            np.array(player_anchors).reshape(-1, 1, 2).astype(np.float32), H
                        ).reshape(-1, 2)
                        
                        # Draw players on court
                        for i, (x_court, y_court) in enumerate(players_court):
                            # Convert court coordinates to pixel coordinates
                            x_px = int(50 + x_court * 20)  # court scale/padding
                            y_px = int(50 + (50 - y_court) * 20)  # flip Y axis
                            
                            # Simple team assignment (could be enhanced)
                            color = self.team_colors["A"] if i % 2 == 0 else self.team_colors["B"]
                            
                            cv2.circle(court_visual, (x_px, y_px), 8, color, -1)
                            cv2.circle(court_visual, (x_px, y_px), 10, (0, 0, 0), 2)
                
                result["court_frame"] = court_visual
                
            except Exception as e:
                print(f"[FixedPipeline] Frame {frame_idx} player processing error: {e}")
        
        return result


def test_fixed_video_pipeline():
    """Test the complete fixed pipeline"""
    print("🧪 Testing Fixed Video Pipeline...")
    
    try:
        from api.src.cv.config import CVConfig
        
        cfg = CVConfig()
        pipeline = FixedVideoPipeline(cfg)
        
        # Test with limited frames
        result = pipeline.process_video(
            video_path=cfg.source_video_path,
            max_frames=50  # Short test
        )
        
        print(f"\n📊 Fixed Pipeline Test Results:")
        print(f"✅ Success: {result.success}")
        print(f"✅ Processed frames: {result.processed_frames}")
        print(f"✅ Successful homographies: {result.successful_homographies}/{result.processed_frames}")
        print(f"✅ Success rate: {result.debug_summary['homography_success_rate']*100:.1f}%")
        print(f"✅ Shots detected: {result.shots_detected}")
        print(f"✅ Processing time: {result.processing_time:.1f}s")
        
        if result.debug_summary['homography_success_rate'] > 0.5:  # >50% success rate
            print("🎉 HOMOGRAPHY FIX SUCCESSFUL!")
            return True
        else:
            print("⚠️  Homography success rate still low, may need further debugging")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fixed_video_pipeline()
    exit(0 if success else 1)
