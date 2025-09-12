#!/usr/bin/env python3
"""
Basketball Computer Vision Utilities
Integrates YOLO, Roboflow, and OpenCV for basketball analysis
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class BasketballConfig:
    """Configuration for basketball detection system."""
    # Model paths
    player_model_path: str = "yolov8m.pt"  # For player detection
    ball_model_path: str = "yolov8m.pt"    # Can use custom ball model
    court_model_path: Optional[str] = None  # Roboflow court detection
    
    # Roboflow settings
    roboflow_api_key: Optional[str] = None
    roboflow_workspace: str = "basketball-formations"
    roboflow_project: str = "basketball-court-detection-2-mlopt"
    roboflow_version: int = 1
    
    # Detection thresholds
    player_confidence: float = 0.5
    ball_confidence: float = 0.3
    court_confidence: float = 0.5
    
    # Tracking settings
    track_players: bool = True
    track_ball: bool = True
    max_tracks: int = 20
    
    # Visualization
    show_labels: bool = True
    show_confidence: bool = True
    box_thickness: int = 2
    text_scale: float = 0.5
    
    # Video processing
    target_fps: int = 30
    resize_width: Optional[int] = 1280
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.roboflow_api_key is None:
            self.roboflow_api_key = os.environ.get('ROBOFLOW_API_KEY')


class BasketballDetector:
    """Main basketball detection and tracking system."""
    
    def __init__(self, config: BasketballConfig = None):
        """Initialize the basketball detector with configuration."""
        self.config = config or BasketballConfig()
        self.models = {}
        self.trackers = {}
        self.court_detector = None
        
        # Initialize models
        self._init_models()
        
        # Initialize trackers
        if self.config.track_players:
            self.trackers['players'] = sv.ByteTrack()
        if self.config.track_ball:
            self.trackers['ball'] = sv.ByteTrack()
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=self.config.box_thickness
        )
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=self.config.box_thickness
        )
    
    def _init_models(self):
        """Initialize YOLO and Roboflow models."""
        try:
            # Load player detection model
            logger.info(f"Loading player model: {self.config.player_model_path}")
            self.models['players'] = YOLO(self.config.player_model_path)
            
            # Load ball detection model (could be same or custom)
            logger.info(f"Loading ball model: {self.config.ball_model_path}")
            self.models['ball'] = YOLO(self.config.ball_model_path)
            
            # Initialize Roboflow court detection if API key available
            if self.config.roboflow_api_key:
                self._init_roboflow()
            else:
                logger.warning("Roboflow API key not set. Court detection disabled.")
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _init_roboflow(self):
        """Initialize Roboflow court detection model."""
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=self.config.roboflow_api_key)
            project = rf.workspace(self.config.roboflow_workspace).project(
                self.config.roboflow_project
            )
            self.court_detector = project.version(self.config.roboflow_version).model
            logger.info("Roboflow court detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Roboflow: {e}")
            self.court_detector = None
    
    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """Detect players in the frame."""
        results = self.models['players'](
            frame, 
            conf=self.config.player_confidence,
            classes=[0],  # Person class in COCO
            verbose=False
        )
        
        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Apply tracking if enabled
        if self.config.track_players and 'players' in self.trackers:
            detections = self.trackers['players'].update_with_detections(detections)
        
        return detections
    
    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        """Detect basketball in the frame."""
        # Using sports ball class (32) from COCO
        # For better results, use a custom basketball model
        results = self.models['ball'](
            frame,
            conf=self.config.ball_confidence,
            classes=[32],  # Sports ball class
            verbose=False
        )
        
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Apply tracking if enabled
        if self.config.track_ball and 'ball' in self.trackers:
            detections = self.trackers['ball'].update_with_detections(detections)
        
        return detections
    
    def detect_court(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect basketball court using Roboflow model."""
        if self.court_detector is None:
            return None
        
        try:
            # Inference on the frame
            result = self.court_detector.predict(
                frame, 
                confidence=self.config.court_confidence
            ).json()
            
            return result
            
        except Exception as e:
            logger.error(f"Court detection failed: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame with all detections."""
        # Resize frame if configured
        if self.config.resize_width:
            height, width = frame.shape[:2]
            aspect_ratio = height / width
            new_height = int(self.config.resize_width * aspect_ratio)
            frame = cv2.resize(frame, (self.config.resize_width, new_height))
        
        # Detect objects
        players = self.detect_players(frame)
        ball = self.detect_ball(frame)
        court = self.detect_court(frame) if self.court_detector else None
        
        # Prepare detection results
        detections = {
            'players': players,
            'ball': ball,
            'court': court,
            'frame_shape': frame.shape
        }
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, detections)
        
        return annotated_frame, detections
    
    def annotate_frame(self, frame: np.ndarray, detections: Dict[str, Any]) -> np.ndarray:
        """Annotate frame with detection results."""
        annotated = frame.copy()
        
        # Annotate players
        if detections['players'] is not None and len(detections['players']) > 0:
            labels = []
            if self.config.show_labels:
                for i, conf in enumerate(detections['players'].confidence):
                    label = f"Player {i+1}"
                    if self.config.show_confidence:
                        label += f" {conf:.2f}"
                    labels.append(label)
            
            annotated = self.box_annotator.annotate(
                scene=annotated,
                detections=detections['players']
            )
            
            if labels:
                annotated = self.label_annotator.annotate(
                    scene=annotated,
                    detections=detections['players'],
                    labels=labels
                )
        
        # Annotate ball
        if detections['ball'] is not None and len(detections['ball']) > 0:
            ball_labels = []
            if self.config.show_labels:
                for conf in detections['ball'].confidence:
                    label = "Ball"
                    if self.config.show_confidence:
                        label += f" {conf:.2f}"
                    ball_labels.append(label)
            
            # Use different color for ball
            ball_annotator = sv.BoxAnnotator(
                thickness=self.config.box_thickness,
                color=sv.Color.YELLOW
            )
            
            annotated = ball_annotator.annotate(
                scene=annotated,
                detections=detections['ball']
            )
            
            if ball_labels:
                annotated = self.label_annotator.annotate(
                    scene=annotated,
                    detections=detections['ball'],
                    labels=ball_labels
                )
        
        # Annotate court (if detected)
        if detections['court'] is not None:
            self._annotate_court(annotated, detections['court'])
        
        return annotated
    
    def _annotate_court(self, frame: np.ndarray, court_data: Dict[str, Any]):
        """Annotate court boundaries on the frame."""
        # This depends on the specific format from Roboflow
        # Typically includes keypoints or polygon boundaries
        try:
            if 'predictions' in court_data:
                for pred in court_data['predictions']:
                    if 'points' in pred:
                        # Draw court boundary
                        points = np.array(pred['points'], dtype=np.int32)
                        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                    elif 'x' in pred and 'y' in pred:
                        # Draw bounding box
                        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Court annotation failed: {e}")
    
    def process_video(self, input_path: str, output_path: str = None,
                     show_preview: bool = False) -> Dict[str, Any]:
        """Process entire video with basketball detection."""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust dimensions if resizing
        if self.config.resize_width:
            aspect_ratio = height / width
            width = self.config.resize_width
            height = int(width * aspect_ratio)
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'avg_players_per_frame': [],
            'ball_detected_frames': 0,
            'court_detected_frames': 0
        }
        
        try:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections = self.process_frame(frame)
                
                # Update statistics
                frame_count += 1
                if detections['players'] is not None:
                    stats['avg_players_per_frame'].append(len(detections['players']))
                if detections['ball'] is not None and len(detections['ball']) > 0:
                    stats['ball_detected_frames'] += 1
                if detections['court'] is not None:
                    stats['court_detected_frames'] += 1
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Show preview if requested
                if show_preview:
                    cv2.imshow('Basketball Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
            
            stats['processed_frames'] = frame_count
            if stats['avg_players_per_frame']:
                stats['avg_players_per_frame'] = np.mean(stats['avg_players_per_frame'])
            else:
                stats['avg_players_per_frame'] = 0
            
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        logger.info(f"Video processing complete. Stats: {stats}")
        return stats


# Utility functions for basketball analysis

def calculate_player_speed(tracks: Dict[int, List[Tuple[int, int]]], 
                          fps: int = 30) -> Dict[int, float]:
    """Calculate player speeds from tracking data."""
    speeds = {}
    for track_id, positions in tracks.items():
        if len(positions) > 1:
            distances = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                distances.append(np.sqrt(dx**2 + dy**2))
            
            # Average speed in pixels per second
            avg_distance = np.mean(distances)
            speeds[track_id] = avg_distance * fps
    
    return speeds


def detect_3_second_violation(player_positions: List[Tuple[int, int]], 
                             paint_area: Tuple[int, int, int, int],
                             fps: int = 30) -> bool:
    """Detect if a player has been in the paint for more than 3 seconds."""
    x1, y1, x2, y2 = paint_area
    time_in_paint = 0
    
    for x, y in player_positions:
        if x1 <= x <= x2 and y1 <= y <= y2:
            time_in_paint += 1
        else:
            time_in_paint = 0
        
        if time_in_paint >= 3 * fps:  # 3 seconds at given FPS
            return True
    
    return False


def calibrate_court_homography(court_keypoints: np.ndarray,
                               reference_court: np.ndarray) -> np.ndarray:
    """Calculate homography matrix for court calibration."""
    if len(court_keypoints) < 4 or len(reference_court) < 4:
        raise ValueError("Need at least 4 points for homography")
    
    # Calculate homography matrix
    H, _ = cv2.findHomography(court_keypoints, reference_court, cv2.RANSAC)
    return H


def transform_to_2d_court(positions: np.ndarray, 
                         homography: np.ndarray) -> np.ndarray:
    """Transform player positions to 2D court coordinates."""
    if positions.shape[1] == 2:
        # Add homogeneous coordinate
        ones = np.ones((positions.shape[0], 1))
        positions = np.hstack([positions, ones])
    
    # Apply homography
    transformed = (homography @ positions.T).T
    
    # Normalize by homogeneous coordinate
    transformed = transformed[:, :2] / transformed[:, 2:3]
    
    return transformed


if __name__ == "__main__":
    # Example usage
    config = BasketballConfig(
        player_confidence=0.5,
        ball_confidence=0.3,
        track_players=True,
        track_ball=True,
        show_labels=True,
        show_confidence=True
    )
    
    detector = BasketballDetector(config)
    
    # Test with a sample image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    annotated, detections = detector.process_frame(test_image)
    
    print(f"Test completed. Detected {len(detections['players'])} players")
    
    # For video processing:
    # stats = detector.process_video("input_video.mp4", "output_video.mp4")
    # print(f"Video stats: {stats}")