# api/src/cv/shot_tracker.py
"""
Shot event tracking module for basketball analysis.
Detects shot attempts, tracks ball trajectory, and determines makes/misses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class ShotEvent:
    """Represents a single shot event with metadata"""
    frame_index: int
    event_type: str  # "START", "MADE", "MISSED"
    shot_type: Optional[str] = None  # "JUMP_SHOT", "LAYUP_DUNK"
    position_image: Optional[Tuple[float, float]] = None
    position_court: Optional[Tuple[float, float]] = None
    distance_ft: Optional[float] = None
    team_id: Optional[int] = None
    confidence: float = 0.0

@dataclass
class ShotEventTracker:
    """
    Tracks basketball shot events across video frames.
    Integrated from the example code with enhancements.
    """
    reset_time_frames: int
    minimum_frames_between_starts: int
    cooldown_frames_after_made: int
    
    # State tracking
    _shot_in_progress: bool = field(default=False, init=False)
    _shot_start_frame: Optional[int] = field(default=None, init=False)
    _last_made_frame: Optional[int] = field(default=None, init=False)
    _last_start_frame: Optional[int] = field(default=None, init=False)
    _consecutive_ball_in_basket: int = field(default=0, init=False)
    _consecutive_jump_shot: int = field(default=0, init=False)
    _consecutive_layup_dunk: int = field(default=0, init=False)
    
    # Configuration
    ball_in_basket_min_consecutive: int = 2
    jump_shot_min_consecutive: int = 3
    layup_dunk_min_consecutive: int = 3
    
    def update(
        self,
        frame_index: int,
        has_jump_shot: bool,
        has_layup_dunk: bool,
        has_ball_in_basket: bool,
        shot_position: Optional[Tuple[float, float]] = None,
    ) -> List[ShotEvent]:
        """
        Update tracker with current frame detections and return any events.
        
        Args:
            frame_index: Current frame number
            has_jump_shot: Whether jump shot detected
            has_layup_dunk: Whether layup/dunk detected
            has_ball_in_basket: Whether ball in basket detected
            shot_position: Optional (x,y) position of shot attempt
            
        Returns:
            List of shot events that occurred in this frame
        """
        events = []
        
        # Update consecutive counters
        self._consecutive_ball_in_basket = (
            self._consecutive_ball_in_basket + 1 if has_ball_in_basket else 0
        )
        self._consecutive_jump_shot = (
            self._consecutive_jump_shot + 1 if has_jump_shot else 0
        )
        self._consecutive_layup_dunk = (
            self._consecutive_layup_dunk + 1 if has_layup_dunk else 0
        )
        
        # Check for shot start
        shot_detected = (
            self._consecutive_jump_shot >= self.jump_shot_min_consecutive or
            self._consecutive_layup_dunk >= self.layup_dunk_min_consecutive
        )
        
        if shot_detected and not self._shot_in_progress:
            # Check if enough time has passed since last start
            can_start = (
                self._last_start_frame is None or
                frame_index - self._last_start_frame >= self.minimum_frames_between_starts
            )
            
            # Check cooldown after made shot
            if self._last_made_frame is not None:
                can_start = can_start and (
                    frame_index - self._last_made_frame >= self.cooldown_frames_after_made
                )
            
            if can_start:
                self._shot_in_progress = True
                self._shot_start_frame = frame_index
                self._last_start_frame = frame_index
                
                shot_type = (
                    "JUMP_SHOT" if self._consecutive_jump_shot >= self.jump_shot_min_consecutive
                    else "LAYUP_DUNK"
                )
                
                events.append(ShotEvent(
                    frame_index=frame_index,
                    event_type="START",
                    shot_type=shot_type,
                    position_image=shot_position,
                    confidence=1.0
                ))
        
        # Check for made shot
        if (self._shot_in_progress and 
            self._consecutive_ball_in_basket >= self.ball_in_basket_min_consecutive):
            
            self._shot_in_progress = False
            self._last_made_frame = frame_index
            
            events.append(ShotEvent(
                frame_index=frame_index,
                event_type="MADE",
                confidence=1.0
            ))
        
        # Check for missed shot (timeout)
        if self._shot_in_progress and self._shot_start_frame is not None:
            if frame_index - self._shot_start_frame >= self.reset_time_frames:
                self._shot_in_progress = False
                
                events.append(ShotEvent(
                    frame_index=frame_index,
                    event_type="MISSED",
                    confidence=0.8
                ))
        
        return events
    
    def reset(self) -> None:
        """Reset all tracking state"""
        self._shot_in_progress = False
        self._shot_start_frame = None
        self._last_made_frame = None
        self._last_start_frame = None
        self._consecutive_ball_in_basket = 0
        self._consecutive_jump_shot = 0
        self._consecutive_layup_dunk = 0


@dataclass
class Shot:
    """Complete shot record with all metadata"""
    x: float  # Court X coordinate (feet)
    y: float  # Court Y coordinate (feet)
    distance: float  # Distance to basket (feet)
    result: bool  # True=made, False=missed
    team: int  # Team ID (0=A, 1=B)
    frame_index: int
    shot_type: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "x": self.x,
            "y": self.y,
            "distance": self.distance,
            "result": self.result,
            "team": self.team,
            "frame_index": self.frame_index,
            "shot_type": self.shot_type,
            "confidence": self.confidence
        }


def extract_made(shots: List[Shot]) -> List[Shot]:
    """Extract only made shots"""
    return [shot for shot in shots if shot.result]


def extract_missed(shots: List[Shot]) -> List[Shot]:
    """Extract only missed shots"""
    return [shot for shot in shots if not shot.result]


def extract_xy(shots: List[Shot]) -> np.ndarray:
    """Extract court coordinates as numpy array"""
    if not shots:
        return np.empty((0, 2), dtype=float)
    return np.array([[shot.x, shot.y] for shot in shots], dtype=float)


def extract_class_id(shots: List[Shot]) -> np.ndarray:
    """Extract team IDs as numpy array"""
    if not shots:
        return np.array([], dtype=int)
    return np.array([shot.team for shot in shots], dtype=int)


def extract_label(shots: List[Shot]) -> List[str]:
    """Extract formatted labels for visualization"""
    return [f"{shot.distance:.1f} ft" for shot in shots]
